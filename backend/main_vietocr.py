import os
import logging
import io
import numpy as np
import uvicorn
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# PaddleOCR chỉ dùng để DETECT vị trí text (nó detect tốt)
# VietOCR dùng để RECOGNIZE text (nó đọc tiếng Việt tốt hơn nhiều)
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCR_SERVER")

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(">>> ĐANG KHỞI TẠO MODEL...")
    try:
        # Chỉ dùng PaddleOCR để DETECT (phát hiện vùng chữ)
        models["detector"] = PaddleOCR(
            use_angle_cls=True,
            lang='vi',
            device='cpu',
            det_limit_side_len=1920,
            det_limit_type='max',
        )
        # VietOCR để RECOGNIZE (đọc nội dung — hỗ trợ đầy đủ dấu tiếng Việt)
        config = Cfg.load_config_from_name('vgg_transformer')  # Model tốt nhất
        config['cnn']['pretrained'] = True
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = True  # BẬT để tăng độ chính xác (đặc biệt với chữ viết tay/chữ có dấu)
        models["recognizer"] = Predictor(config)

        logger.info(">>> KHỞI TẠO THÀNH CÔNG!")
    except Exception as e:
        logger.error(f">>> LỖI: {e}")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def resize_image(img_np: np.ndarray) -> np.ndarray:
    h, w = img_np.shape[:2]
    if h < 800 or w < 800:
        scale = max(800 / h, 800 / w)
        img_np = cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img_np

def enhance_image(img_np: np.ndarray) -> np.ndarray:
    # MẸO: Trích xuất kênh Blue để tàng hình các dòng kẻ tập vở màu xanh
    # và chỉ giữ lại nét mực đen/tím đậm.
    blue_channel = img_np[:, :, 2]
    
    # Dùng CLAHE tăng mạnh tương phản trên ảnh đã mất dòng kẻ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blue_channel)
    
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


@app.post("/api/extract-text")
async def extract_text(image: UploadFile = File(...)):
    if "detector" not in models or "recognizer" not in models:
        return {"status": "error", "message": "Model chưa sẵn sàng."}

    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        # FIX EXIF ROTATION: Ảnh chụp từ điện thoại lưu metadata xoay (90°, 270°...)
        # PIL mặc định KHÔNG đọc EXIF → ảnh bị nằm ngang khi xử lý → OCR thất bại
        # exif_transpose() tự động xoay đúng hướng theo EXIF trước khi xử lý
        pil_img = ImageOps.exif_transpose(pil_img)

        img_np = np.array(pil_img)
        original_h, original_w = img_np.shape[:2]

        resized_img = resize_image(img_np.copy())
        processed_img = enhance_image(resized_img.copy())
        
        processed_h, processed_w = processed_img.shape[:2]
        scale_x = original_w / processed_w
        scale_y = original_h / processed_h

        # Bước 1: Dùng PaddleOCR chỉ để DETECT vùng chữ
        det_result = models["detector"].ocr(processed_img, rec=False, cls=False)

        full_text = ""
        details = []

        if det_result and det_result[0]:
            # RECOGNITION image has natural contrast (no sharpen/CLAHE garbage)
            pil_resized = Image.fromarray(resized_img)

            # Thuật toán gom dòng và sắp xếp chính xác:
            boxes = det_result[0]
            box_data = []
            for b in boxes:
                pts = np.array(b)
                y_min = pts[:, 1].min()
                y_max = pts[:, 1].max()
                x_min = pts[:, 0].min()
                x_max = pts[:, 0].max()
                y_center = (y_min + y_max) / 2
                height = y_max - y_min
                box_data.append({
                    "box": b,
                    "y_center": y_center,
                    "x_min": x_min,
                    "y_min": y_min,
                    "height": height
                })

            # 1. Sắp xếp sơ bộ từ trên xuống dưới theo tâm Y
            box_data.sort(key=lambda x: x["y_center"])

            lines = []
            if box_data:
                current_line = [box_data[0]]
                # 2. Gom thành từng dòng nếu tâm Y của box liền kề lệch nhau không quá nửa chiều cao chữ
                for i in range(1, len(box_data)):
                    item = box_data[i]
                    prev_item = current_line[-1]
                    avg_height = (item["height"] + prev_item["height"]) / 2
                    
                    if abs(item["y_center"] - prev_item["y_center"]) < avg_height * 0.5:
                        current_line.append(item)
                    else:
                        lines.append(current_line)
                        current_line = [item]
                if current_line:
                    lines.append(current_line)

            # 3. Quét từng dòng từ trái qua phải và ghép chữ
            for line in lines:
                line.sort(key=lambda x: x["x_min"])
                
                line_text = ""
                for item in line:
                    box = item["box"]
                    pts = np.array(box, dtype=np.int32)
                    x_min = max(0, pts[:, 0].min())
                    y_min = max(0, pts[:, 1].min())
                    x_max = min(processed_w, pts[:, 0].max())
                    y_max = min(processed_h, pts[:, 1].max())

                    box_w = x_max - x_min
                    box_h = y_max - y_min

                    # LỌC 1 — ASPECT RATIO: Bỏ qua box gần vuông/cao hơn rộng
                    # Vùng chữ thật luôn rộng hơn cao (tỉ lệ ngang).
                    # Box gần vuông hoặc cao hơn rộng = hình ảnh, icon, logo → bỏ qua
                    if box_h > 0 and (box_w / box_h) < 1.2:
                        logger.debug(f"Bỏ qua box vuông/đứng: w={box_w} h={box_h} ratio={box_w/box_h:.2f}")
                        continue

                    # TRÁNH BỎ SÓT: Cắt rộng thêm 1 chút để lấy dấu, nhưng không quá lớn làm nhiễu
                    pad_x = 3
                    pad_y_top = 4
                    pad_y_bottom = 3
                    x_min_pad = max(0, x_min - pad_x)
                    y_min_pad = max(0, y_min - pad_y_top)
                    x_max_pad = min(processed_w, x_max + pad_x)
                    y_max_pad = min(processed_h, y_max + pad_y_bottom)

                    cropped = pil_resized.crop((x_min_pad, y_min_pad, x_max_pad, y_max_pad))

                    # Bước 2: Nhận diện với VietOCR
                    text, conf = models["recognizer"].predict(cropped, return_prob=True)
                    text = text.strip()

                    # LỌC 2 — CONFIDENCE: Bỏ qua nếu VietOCR không tự tin
                    # Ngưỡng 0.45 (nếu có): đủ chặt để loại nhiễu, đủ mềm để nhận dạng chữ nghiêng/mờ
                    if conf is not None and conf < 0.45:
                        logger.debug(f"Bỏ qua low-conf: '{text}' conf={conf:.3f}")
                        continue

                    # LỌC 3 — TEXT RÁC: Bỏ qua kết quả ngắn không mang nghĩa tiếng Việt
                    # Khi VietOCR đọc hình ảnh/biểu đồ → ra ký tự rác như "0", "go", "c", "|"
                    if len(text) <= 2:
                        # Chỉ chấp nhận nếu chứa ít nhất 1 chữ cái và không phải ký tự đặc biệt
                        import re as _re
                        if not _re.search(r'[a-zA-ZÀ-ỹ]', text):
                            logger.debug(f"Bỏ qua text rác ngắn: '{text}'")
                            continue

                    if text:
                        # Gom khoảng trắng với các cụm trong cùng dòng
                        if line_text:
                            line_text += " " + text
                        else:
                            line_text = text

                        # Original box coordinate mapping
                        original_box = [
                            [int(pt[0] * scale_x), int(pt[1] * scale_y)]
                            for pt in box
                        ]
                        details.append({
                            "box": original_box,
                            "text": text,
                            "confidence": float(conf) if conf is not None else 1.0
                        })

                if line_text:
                    full_text += line_text + "\n"

        logger.info(f"Nhận diện được {len(details)} từ/cụm cụm.")

        return {
            "status": "success",
            "text": full_text.strip(),
            "details": details,
            "imageSize": {"width": original_w, "height": original_h}
        }
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
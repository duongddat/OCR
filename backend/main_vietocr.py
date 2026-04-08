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
    # Chuyển sang grayscale thay vì chỉ dùng kênh Blue để không làm biến mất nét mực xanh nhạt
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Dùng CLAHE tăng mạnh tương phản trên ảnh 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
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
            # Ảnh thuần RGB giữ nguyên màu sắc, tránh dùng kênh Blue làm mất mực nhạt 
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
                # 2. Gom thành từng dòng thay vì nối đuôi theo phần tử liền trước
                for b in box_data:
                    placed = False
                    for line in lines:
                        line_y_center = sum(item["y_center"] for item in line) / len(line)
                        line_height = sum(item["height"] for item in line) / len(line)
                        
                        # So sánh b["y_center"] với trung bình của toàn dòng
                        # Giảm ngưỡng từ 0.4 xuống 0.25 để tách các dòng liền nhau ra
                        if abs(b["y_center"] - line_y_center) < line_height * 0.25:
                            line.append(b)
                            placed = True
                            break
                    if not placed:
                        lines.append([b])
            
            # Sắp xếp các dòng từ trên xuống dưới theo độ cao trung bình
            lines.sort(key=lambda l: sum(item["y_center"] for item in l) / len(l))

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

                    # TRÁNH BỎ SÓT: Tăng pad_x lớn để bao gồm trọn các dấu ngoặc ( ) của chữ nghiêng
                    pad_x = 12
                    pad_y_top = 8
                    pad_y_bottom = 5
                    x_min_pad = max(0, x_min - pad_x)
                    y_min_pad = max(0, y_min - pad_y_top)
                    x_max_pad = min(processed_w, x_max + pad_x)
                    y_max_pad = min(processed_h, y_max + pad_y_bottom)

                    cropped = pil_resized.crop((x_min_pad, y_min_pad, x_max_pad, y_max_pad))

                    # Bước 2: Nhận diện với VietOCR
                    text, conf = models["recognizer"].predict(cropped, return_prob=True)
                    text = text.strip()
                    
                    # FIX: VietOCR hay bị lỗi trộn chữ HOA với dấu thường (VD: "NGHIệP NGHèO")
                    # Nếu một từ có phần lớn là in hoa, ta sẽ ép nó thành in hoa chuẩn (VD: NGHIỆP NGHÈO)
                    fixed_words = []
                    for w in text.split():
                        alphas = [c for c in w if c.isalpha()]
                        if len(alphas) >= 2:
                            uppers = sum(1 for c in alphas if c.isupper())
                            if uppers >= 2 and uppers >= len(alphas) / 2:
                                fixed_words.append(w.upper())
                                continue
                        fixed_words.append(w)
                    text = " ".join(fixed_words)

                    # LỌC 2 — CONFIDENCE: Bỏ qua nếu VietOCR quá thiếu tự tin 
                    if conf is not None and conf < 0.15:
                        continue

                    # LỌC CHỐNG NHIỄU GIẢ: Các vệt dòng kẻ tập trống rỗng thường có w/h rất lớn
                    # và hay bị VietOCR nhận diện ảo thành "1", "l", "-", "|"
                    if text in ["1", "l", "I", "|", "!", "-", "'", "`"] and box_h > 0 and (box_w / box_h) > 2.0:
                        logger.debug(f"Bỏ qua dòng kẻ rác bị nhận thành 1: '{text}' w/h={box_w/box_h:.2f}")
                        continue

                    # LỌC 3 — TEXT RÁC: Giữ lại dấu câu và chữ số 
                    if len(text) <= 2:
                        import re as _re
                        if not _re.search(r'[a-zA-ZÀ-ỹ0-9.,;:!?()\[\]]', text):
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
import os
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
import logging
import io
import re
import numpy as np
import uvicorn
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCR_SERVER")

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(">>> ĐANG KHỞI TẠO MODEL...")
    try:
        models["detector"] = PaddleOCR(
            use_angle_cls=True,
            lang='vi',
            device='cpu',
            enable_mkldnn=False,
            det=True,       # Đảm bảo chỉ dùng Detection
            rec=False,      # BẮT BUỘC Tắt Recognition của PaddleOCR để tăng tốc X3 lần
            cls=False,
            det_limit_side_len=1920,
            det_limit_type='max',
        )
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = True
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = True
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
    # Đã tắt màng lọc kênh Blue do bộ lọc này phá hủy màu mực xanh và làm nhiễu tài liệu chuẩn.
    # Thay vào đó chỉ tăng độ tương phản nhẹ và giữ nguyên không gian màu chuẩn cho PaddleOCR.
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    
    # Chuyển sang ảnh xám để tăng tương phản, sau đó convert lại RGB
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def extract_polys_from_result(det_result) -> list:
    """
    Chuẩn hóa output PaddleOCR về list polygon [[x,y],...].

    PaddleOCR 2.x:
        det_result[0] = list box  →  [ [[[x,y],...], ('text', conf)], ... ]

    PaddleOCR 3.x:
        det_result[0] = dict      →  {'dt_polys': [...], 'dt_scores': [...], ...}
    """
    if not det_result:
        return []

    page = det_result[0]

    # ── PaddleOCR 3.x ─────────────────────────────────────────────────────────
    if isinstance(page, dict):
        raw_polys = page.get('dt_polys', [])
        polys = []
        for poly in raw_polys:
            pts = np.array(poly)
            if pts.ndim == 2 and pts.shape[-1] == 2:
                polys.append(pts.tolist())
        logger.info(f"[PaddleOCR 3.x] Detect {len(polys)} vùng chữ")
        return polys

    # ── PaddleOCR 2.x ─────────────────────────────────────────────────────────
    if isinstance(page, list):
        polys = []
        for b in page:
            if (isinstance(b, (list, tuple))
                    and len(b) == 2
                    and isinstance(b[1], (tuple, list))
                    and len(b[1]) == 2
                    and isinstance(b[1][0], str)):
                b_points = b[0]
            elif (isinstance(b, (list, tuple))
                  and len(b) > 0
                  and isinstance(b[0], (list, tuple))):
                b_points = b
            else:
                continue
            pts = np.array(b_points)
            if pts.ndim == 2 and pts.shape[-1] == 2:
                polys.append(pts.tolist())
        logger.info(f"[PaddleOCR 2.x] Detect {len(polys)} vùng chữ")
        return polys

    logger.warning(f"Format PaddleOCR không xác định: type={type(page)}")
    return []


@app.post("/api/extract-text")
async def extract_text(image: UploadFile = File(...)):
    if "detector" not in models or "recognizer" not in models:
        return {"status": "error", "message": "Model chưa sẵn sàng."}

    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        pil_img = ImageOps.exif_transpose(pil_img)

        img_np = np.array(pil_img)
        original_h, original_w = img_np.shape[:2]

        resized_img = resize_image(img_np.copy())
        processed_img = enhance_image(resized_img.copy())

        processed_h, processed_w = processed_img.shape[:2]
        scale_x = original_w / processed_w
        scale_y = original_h / processed_h

        # Bước 1: Detect vùng chữ
        det_result = models["detector"].ocr(processed_img)
        logger.info(f"det_result type={type(det_result)}, len={len(det_result) if det_result else 0}")

        # ✅ FIX CHÍNH: chuẩn hóa output PaddleOCR 3.x
        polys = extract_polys_from_result(det_result)

        full_text = ""
        details = []

        if polys:
            pil_resized = Image.fromarray(resized_img)

            # Tính metadata để sắp xếp dòng
            box_data = []
            for poly in polys:
                pts = np.array(poly)
                y_min    = float(pts[:, 1].min())
                y_max    = float(pts[:, 1].max())
                x_min    = float(pts[:, 0].min())
                y_center = (y_min + y_max) / 2
                height   = y_max - y_min
                box_data.append({
                    "box":      poly,
                    "y_center": y_center,
                    "x_min":    x_min,
                    "y_min":    y_min,
                    "height":   height,
                })

            box_data.sort(key=lambda x: x["y_center"])

            # Gom dòng
            lines = []
            current_line = [box_data[0]]
            for i in range(1, len(box_data)):
                item      = box_data[i]
                prev_item = current_line[-1]
                avg_h     = (item["height"] + prev_item["height"]) / 2
                if abs(item["y_center"] - prev_item["y_center"]) < avg_h * 0.5:
                    current_line.append(item)
                else:
                    lines.append(current_line)
                    current_line = [item]
            lines.append(current_line)

            # Nhận diện từng dòng trái → phải
            for line in lines:
                line.sort(key=lambda x: x["x_min"])
                line_text = ""

                for item in line:
                    box = item["box"]
                    pts = np.array(box, dtype=np.int32)

                    x_min_i = max(0, int(pts[:, 0].min()))
                    y_min_i = max(0, int(pts[:, 1].min()))
                    x_max_i = min(processed_w, int(pts[:, 0].max()))
                    y_max_i = min(processed_h, int(pts[:, 1].max()))

                    box_w = x_max_i - x_min_i
                    box_h = y_max_i - y_min_i

                    # LỌC 1 — ASPECT RATIO
                    if box_h > 0 and (box_w / box_h) < 1.2:
                        logger.debug(f"Bỏ qua box vuông/đứng ratio={box_w/box_h:.2f}")
                        continue

                    # Padding nhỏ
                    x1 = max(0, x_min_i - 3)
                    y1 = max(0, y_min_i - 4)
                    x2 = min(processed_w, x_max_i + 3)
                    y2 = min(processed_h, y_max_i + 3)

                    cropped = pil_resized.crop((x1, y1, x2, y2))

                    # Bước 2: Nhận diện với VietOCR
                    # Đưa vào Try-Catch vì VietOCR có thể raise KeyError với các ký tự ngoại ngữ lạ
                    try:
                        text, conf = models["recognizer"].predict(cropped, return_prob=True)
                        text = text.strip()
                    except Exception as e:
                        logger.warning(f"VietOCR lỗi nhận diện (có thể do ký tự lạ): {e}")
                        text, conf = "", 0.0

                    # Fix HOA lẫn thường
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

                    # LỌC 2 — CONFIDENCE
                    if conf is not None and conf < 0.45:
                        logger.debug(f"Bỏ qua low-conf: '{text}' conf={conf:.3f}")
                        continue

                    # LỌC 3 — TEXT RÁC
                    if len(text) <= 2 and not re.search(r'[a-zA-ZÀ-ỹ]', text):
                        logger.debug(f"Bỏ qua text rác: '{text}'")
                        continue

                    if text:
                        line_text = (line_text + " " + text).strip() if line_text else text
                        original_box = [
                            [int(pt[0] * scale_x), int(pt[1] * scale_y)]
                            for pt in box
                        ]
                        details.append({
                            "box":        original_box,
                            "text":       text,
                            "confidence": float(conf) if conf is not None else 1.0,
                        })

                if line_text:
                    full_text += line_text + "\n"

        logger.info(f"Nhận diện được {len(details)} từ/cụm.")
        return {
            "status":    "success",
            "text":      full_text.strip(),
            "details":   details,
            "imageSize": {"width": int(original_w), "height": int(original_h)},
        }

    except Exception as e:
        logger.exception(f"Lỗi xử lý ảnh: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
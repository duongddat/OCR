import os
import logging
import io
import asyncio
import numpy as np
import uvicorn
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter, ImageEnhance

os.environ["GLOG_minloglevel"] = "2"
import warnings
warnings.filterwarnings("ignore")

from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCR_SERVER")

models = {}
ocr_semaphore = asyncio.Semaphore(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(">>> ĐANG KHỞI TẠO PADDLEOCR...")
    try:
        models["ocr"] = PaddleOCR(
            use_angle_cls=True,         # ✅ Bật classifier góc chữ
            lang='vi',
            device='cpu',
            # ✅ Dùng model chất lượng cao hơn (PP-OCRv4)
            det_model_dir=None,         # Dùng model detect mặc định
            rec_model_dir=None,         # Dùng model rec mặc định
            ocr_version='PP-OCRv4',     # ✅ Version mới nhất, tốt hơn nhiều
            rec_char_dict_path=None,
            use_space_char=True,        # ✅ Giữ khoảng trắng
            # ✅ Tăng độ phân giải xử lý nội bộ
            det_limit_side_len=1920,    # Tăng từ 960 lên 1920
            det_limit_type='max',
            rec_batch_num=6,
            # ✅ Ngưỡng tin cậy
            drop_score=0.3,             # Giảm threshold để giữ lại ký tự có dấu
        )
        logger.info(">>> KHỞI TẠO MODEL THÀNH CÔNG!")
    except Exception as e:
        logger.error(f">>> LỖI KHỞI TẠO: {e}")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def preprocess_image(img_np: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý ảnh để tăng độ chính xác OCR tiếng Việt.
    """
    # 1. Upscale nếu ảnh quá nhỏ (OCR cần tối thiểu ~30px chiều cao chữ)
    h, w = img_np.shape[:2]
    if h < 800 or w < 800:
        scale = max(800 / h, 800 / w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.info(f"Upscaled: {w}x{h} → {new_w}x{new_h}")

    # 2. Chuyển sang grayscale để xử lý
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 3. Tăng độ tương phản bằng CLAHE (tốt hơn equalizeHist cho ảnh chụp)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Khử nhiễu nhẹ (giữ nét chữ)
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 5. Sharpen để làm rõ dấu tiếng Việt
    kernel_sharpen = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    gray = cv2.filter2D(gray, -1, kernel_sharpen)

    # 6. Chuyển lại RGB để PaddleOCR xử lý
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return result


def _extract_text_sync(contents: bytes):
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)

        original_h, original_w = img_np.shape[:2]

        # ✅ Tiền xử lý ảnh trước khi đưa vào OCR
        processed_img = preprocess_image(img_np)

        # ✅ Tính scale để map tọa độ về ảnh gốc
        processed_h, processed_w = processed_img.shape[:2]
        scale_x = original_w / processed_w
        scale_y = original_h / processed_h

        result = models["ocr"].ocr(processed_img, cls=True)  # cls=True vì đã bật use_angle_cls

        full_text = ""
        details = []

        if result and len(result) > 0 and result[0]:
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                conf = float(line[1][1])

                # ✅ Scale tọa độ box về kích thước ảnh gốc
                original_box = [
                    [int(pt[0] * scale_x), int(pt[1] * scale_y)]
                    for pt in box
                ]

                full_text += text + "\n"
                details.append({
                    "box": original_box,
                    "text": text,
                    "confidence": conf
                })

        logger.info(f"Nhận diện được {len(details)} dòng văn bản.")

        return {
            "status": "success",
            "text": full_text.strip(),
            "details": details,
            "imageSize": {
                "width": original_w,
                "height": original_h
            }
        }
    except Exception as e:
        logger.error(f"Lỗi khi nhận giải mã: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/extract-text")
async def extract_text(image: UploadFile = File(...)):
    if "ocr" not in models:
        return {"status": "error", "message": "Model chưa sẵn sàng."}

    try:
        contents = await image.read()
        
        # Đẩy logic nặng xuống Thread pool chờ Semaphore để không làm treo Server FASTAPI
        async with ocr_semaphore:
            result_data = await asyncio.to_thread(_extract_text_sync, contents)
            
        return result_data
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
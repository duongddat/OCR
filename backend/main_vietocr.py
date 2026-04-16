import os
import logging
import io
import asyncio
import re
import fitz  # PyMuPDF
import numpy as np
import uvicorn
import cv2
import torch
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCR_SERVER")

models = {}
ocr_semaphore = asyncio.Semaphore(1)

_RECOGNITION_WORKERS = int(os.environ.get("OCR_RECOGNITION_WORKERS", "3"))
_TORCH_TOTAL_THREADS  = torch.get_num_threads()
_TORCH_PER_WORKER     = max(1, _TORCH_TOTAL_THREADS // _RECOGNITION_WORKERS)
_DET_LIMIT   = int(os.environ.get("OCR_DET_LIMIT",   "960"))
_RECOG_LIMIT = int(os.environ.get("OCR_RECOG_LIMIT", "1280"))

_recognition_pool = ThreadPoolExecutor(
    max_workers=_RECOGNITION_WORKERS,
    thread_name_prefix="vietocr",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(">>> ĐANG KHỞI TẠO MODEL...")
    try:
        models["detector"] = PaddleOCR(
            use_angle_cls=False,
            lang='vi',
            device='cpu',
            det_limit_side_len=_DET_LIMIT,
            det_limit_type='max',
        )
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = True
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        models["recognizer"] = Predictor(config)
        _warmup_vietocr(models["recognizer"])
        logger.info(">>> KHỞI TẠO THÀNH CÔNG!")
    except Exception as e:
        logger.error(f">>> LỖI: {e}")
    yield
    _recognition_pool.shutdown(wait=False)
    models.clear()


def _warmup_vietocr(recognizer):
    dummy = Image.fromarray(np.ones((32, 128, 3), dtype=np.uint8) * 200)
    try:
        recognizer.predict(dummy, return_prob=True)
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def resize_to_limit(img_np: np.ndarray, limit: int) -> np.ndarray:
    h, w = img_np.shape[:2]
    longest = max(h, w)
    if longest == limit:
        return img_np
    scale = limit / longest
    interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=interp)


def enhance_image(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if gray.std() > 55.0:
        return img_np
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# NMS + noise filter
# ─────────────────────────────────────────────────────────────────────────────

def _get_rect(box_pts) -> tuple[float, float, float, float]:
    pts = np.array(box_pts)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), \
           float(pts[:, 0].max()), float(pts[:, 1].max())


def _nms_and_filter_boxes(box_data: list[dict], img_w: int, img_h: int) -> list[dict]:
    if not box_data:
        return []

    img_area = img_w * img_h

    for b in box_data:
        r = _get_rect(b["box"])
        b["_rect"]  = r
        b["_area"]  = (r[2] - r[0]) * (r[3] - r[1])

    min_abs = img_area * 0.0005
    box_data = [b for b in box_data if b["_area"] >= min_abs]
    if not box_data:
        return []

    areas_list = [b["_area"] for b in box_data]
    median_area = np.median(areas_list)
    min_rel = median_area * 0.08
    box_data = [b for b in box_data if b["_area"] >= min_rel]
    if not box_data:
        return []

    sorted_boxes = sorted(box_data, key=lambda b: b["_area"], reverse=True)

    rects = np.array([b["_rect"] for b in sorted_boxes])
    areas = np.array([b["_area"] for b in sorted_boxes])

    keep_indices = []
    order = np.arange(len(sorted_boxes))

    while order.size > 0:
        i = order[0]
        keep_indices.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(rects[i, 0], rects[order[1:], 0])
        yy1 = np.maximum(rects[i, 1], rects[order[1:], 1])
        xx2 = np.minimum(rects[i, 2], rects[order[1:], 2])
        yy2 = np.minimum(rects[i, 3], rects[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        ios = inter / areas[order[1:]]

        inds = np.where((iou < 0.40) & (ios < 0.60))[0]
        order = order[inds + 1]

    result = [sorted_boxes[idx] for idx in keep_indices]
    logger.debug("NMS Vectorized: %d → %d boxes", len(box_data), len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Smart crop
# ─────────────────────────────────────────────────────────────────────────────

def _smart_crop_numpy(img_np: np.ndarray,
                      x_min: float, y_min: float,
                      x_max: float, y_max: float,
                      box_h: float) -> list[Image.Image]:
    h, w = img_np.shape[:2]

    x1_full = max(0, int(x_min - 12))
    y1_full = max(0, int(y_min -  8))
    x2_full = min(w, int(x_max + 12))
    y2_full = min(h, int(y_max +  5))

    crops = [Image.fromarray(img_np[y1_full:y2_full, x1_full:x2_full])]

    if box_h > 40:
        margin_x = int((x_max - x_min) * 0.08)
        margin_y = int(box_h * 0.10)
        x1_tight = max(0, int(x_min + margin_x))
        y1_tight = max(0, int(y_min + margin_y))
        x2_tight = min(w, int(x_max - margin_x))
        y2_tight = min(h, int(y_max - margin_y))

        if x2_tight > x1_tight + 5 and y2_tight > y1_tight + 5:
            crops.append(Image.fromarray(img_np[y1_tight:y2_tight, x1_tight:x2_tight]))

    return crops


# ─────────────────────────────────────────────────────────────────────────────
# Line grouping
# ─────────────────────────────────────────────────────────────────────────────

def _group_boxes_into_lines(box_data: list[dict]) -> list[list[dict]]:
    if not box_data:
        return []
    y_centers = np.array([b["y_center"] for b in box_data])
    heights   = np.array([b["height"]   for b in box_data])
    order     = np.argsort(y_centers, kind='stable')
    sorted_boxes    = [box_data[i] for i in order]
    sorted_ycenters = y_centers[order]
    sorted_heights  = heights[order]

    lines, y_sums, h_sums, counts = [], [], [], []
    for i, b in enumerate(sorted_boxes):
        yc, h = sorted_ycenters[i], sorted_heights[i]
        placed = False
        for j in range(len(lines)):
            if abs(yc - y_sums[j] / counts[j]) < (h_sums[j] / counts[j]) * 0.25:
                lines[j].append(b)
                y_sums[j] += yc; h_sums[j] += h; counts[j] += 1
                placed = True
                break
        if not placed:
            lines.append([b]); y_sums.append(yc); h_sums.append(h); counts.append(1)

    means = [y_sums[j] / counts[j] for j in range(len(lines))]
    return [lines[j] for j in np.argsort(means, kind='stable')]


def _prepare_valid_crops(
    lines, recog_img_np, det_w, det_h, recog_w, recog_h, sx_d2r, sy_d2r
) -> list[dict]:
    valid = []
    for line_idx, line in enumerate(lines):
        for box_idx, item in enumerate(sorted(line, key=lambda x: x["x_min"])):
            box = item["box"]
            pts = np.array(box, dtype=np.float32)

            x_min_d = max(0.0, pts[:, 0].min())
            y_min_d = max(0.0, pts[:, 1].min())
            x_max_d = min(float(det_w), pts[:, 0].max())
            y_max_d = min(float(det_h), pts[:, 1].max())

            bw = x_max_d - x_min_d
            bh = y_max_d - y_min_d
            if bh <= 0 or bw <= 0 or (bw / bh) < 1.2:
                continue

            x_min_r = x_min_d * sx_d2r
            y_min_r = y_min_d * sy_d2r
            x_max_r = x_max_d * sx_d2r
            y_max_r = y_max_d * sy_d2r
            bh_r    = bh * sy_d2r

            crops = _smart_crop_numpy(
                recog_img_np,
                x_min_r, y_min_r, x_max_r, y_max_r,
                bh_r
            )

            valid.append({
                "crops":    crops,
                "box":      box,
                "line_idx": line_idx,
                "box_idx":  box_idx,
                "box_w":    int(bw),
                "box_h":    int(bh),
            })
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Batch VietOCR
# ─────────────────────────────────────────────────────────────────────────────

def _batch_vietocr_predict(crops: list[Image.Image]) -> list[tuple[str, float]]:
    recognizer = models["recognizer"]
    if not crops:
        return []
    if len(crops) == 1:
        t, c = recognizer.predict(crops[0], return_prob=True)
        return [(t.strip(), float(c) if c else 1.0)]
    try:
        model = recognizer.model
        transform = recognizer.transformers

        tensors = [transform(c) for c in crops]
        max_w = max(t.shape[2] for t in tensors)

        batch = torch.zeros(len(tensors), 3, 32, max_w)
        for i, t in enumerate(tensors):
            batch[i, :, :, :t.shape[2]] = t

        with torch.inference_mode():
            src_batch = model.cnn(batch)

        results = []
        for i in range(len(crops)):
            src_i = src_batch[i:i+1]
            with torch.inference_mode():
                if recognizer.config['predictor']['beamsearch']:
                    sent = model.beamsearch(src_i)
                    text = recognizer.vocab.decode(sent)
                    prob = 1.0
                else:
                    s, prob = model.translate(src_i)
                    text = recognizer.vocab.decode(s[0].tolist())
                    prob = float(prob) if prob is not None else 1.0
            results.append((text.strip(), prob))
        return results
    except Exception as e:
        logger.debug("Batch CNN fallback sequential: %s", e)
        out = []
        for c in crops:
            t, conf = recognizer.predict(c, return_prob=True)
            out.append((t.strip(), float(conf) if conf else 1.0))
        return out


def _recognize_chunk(crops_chunk: list[Image.Image]) -> list[tuple[str, float]]:
    torch.set_num_threads(_TORCH_PER_WORKER)
    return _batch_vietocr_predict(crops_chunk)


def _run_parallel_chunked_recognition(valid_crops: list[dict]) -> list[tuple[str, float]]:
    if not valid_crops:
        return []

    all_crops   = []
    item_ranges = []
    for item in valid_crops:
        start = len(all_crops)
        all_crops.extend(item["crops"])
        item_ranges.append((start, len(all_crops)))

    n = len(all_crops)
    if n <= 5:
        all_results = _batch_vietocr_predict(all_crops)
    else:
        chunk_size  = max(1, -(-n // _RECOGNITION_WORKERS))
        chunks      = [all_crops[i:i+chunk_size] for i in range(0, n, chunk_size)]
        futures     = {
            _recognition_pool.submit(_recognize_chunk, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        flat_chunks = [None] * len(chunks)
        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                flat_chunks[chunk_idx] = future.result()
            except Exception as e:
                logger.warning("Chunk %d failed: %s", chunk_idx, e)
                flat_chunks[chunk_idx] = [("", 0.0)] * len(chunks[chunk_idx])
        all_results = [r for chunk in flat_chunks for r in chunk]

    best_results = []
    for start, end in item_ranges:
        candidates = all_results[start:end]
        best = max(candidates, key=lambda x: x[1])
        best_results.append(best)

    return best_results


# ─────────────────────────────────────────────────────────────────────────────
# Confidence filter
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_CHARS = frozenset(["1", "l", "I", "|", "!", "-", "'", "`"])


def _fix_mixed_case(text: str) -> str:
    fixed = []
    for w in text.split():
        alphas = [c for c in w if c.isalpha()]
        if len(alphas) >= 2:
            uppers = sum(1 for c in alphas if c.isupper())
            if uppers >= 2 and uppers >= len(alphas) / 2:
                fixed.append(w.upper()); continue
        fixed.append(w)
    return " ".join(fixed)


def _is_valid_text(text: str, box_w: int, box_h: int, conf: float) -> bool:
    text_len = len(text)

    if text_len == 1 and text.isalpha() and text.isupper():
        if conf < 0.70:
            return False
    elif text_len <= 2:
        if conf < 0.40:
            return False
        if not re.search(r'[a-zA-ZÀ-ỹ0-9.,;:!?()\[\]]', text):
            return False
    else:
        if conf < 0.15:
            return False

    if text in _NOISE_CHARS and box_h > 0 and (box_w / box_h) > 2.0:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline — nhận numpy array (dùng chung cho ảnh và PDF)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_from_numpy(img_np: np.ndarray) -> dict:
    """
    Pipeline OCR hoàn chỉnh từ numpy array RGB uint8.
    Tách ra để PDF có thể gọi trực tiếp, tránh encode/decode PNG thừa.
    """
    try:
        original_h, original_w = img_np.shape[:2]

        det_img  = resize_to_limit(img_np, _DET_LIMIT)
        det_img  = enhance_image(det_img)
        det_h, det_w = det_img.shape[:2]

        recog_img = resize_to_limit(img_np, _RECOG_LIMIT)
        recog_h, recog_w = recog_img.shape[:2]

        sx_d2r = recog_w / det_w
        sy_d2r = recog_h / det_h
        sx_d2o = original_w / det_w
        sy_d2o = original_h / det_h

        # ── PHASE 1: Detection ───────────────────────────────────────────────
        det_result = models["detector"].ocr(det_img, rec=False, cls=False)

        if not det_result or not det_result[0]:
            return {"status": "success", "text": "", "details": [],
                    "imageSize": {"width": original_w, "height": original_h}}

        # ── PHASE 2: Box metadata ────────────────────────────────────────────
        raw_boxes = det_result[0]
        box_data  = []
        for b in raw_boxes:
            pts   = np.array(b)
            y_min = pts[:, 1].min(); y_max = pts[:, 1].max()
            box_data.append({
                "box":      b,
                "y_center": (y_min + y_max) / 2,
                "x_min":    pts[:, 0].min(),
                "y_min":    y_min,
                "height":   y_max - y_min,
            })

        logger.info("PaddleOCR raw: %d boxes", len(box_data))

        # ── PHASE 2.5: NMS + noise filter ───────────────────────────────────
        box_data = _nms_and_filter_boxes(box_data, det_w, det_h)
        logger.info("After NMS: %d boxes", len(box_data))

        if not box_data:
            return {"status": "success", "text": "", "details": [],
                    "imageSize": {"width": original_w, "height": original_h}}

        # ── PHASE 3: Line grouping ───────────────────────────────────────────
        lines = _group_boxes_into_lines(box_data)

        # ── PHASE 4: Smart crop ──────────────────────────────────────────────
        valid_crops = _prepare_valid_crops(
            lines, recog_img,
            det_w, det_h, recog_w, recog_h,
            sx_d2r, sy_d2r,
        )

        if not valid_crops:
            return {"status": "success", "text": "", "details": [],
                    "imageSize": {"width": original_w, "height": original_h}}

        # ── PHASE 5: Parallel batch recognition ─────────────────────────────
        recog_results = _run_parallel_chunked_recognition(valid_crops)

        # ── PHASE 6: Post-process ────────────────────────────────────────────
        line_texts: dict[int, list[tuple[int, str]]] = {}
        details = []

        for item, (raw_text, conf) in zip(valid_crops, recog_results):
            if not raw_text:
                continue
            text = _fix_mixed_case(raw_text)
            if not _is_valid_text(text, item["box_w"], item["box_h"], conf):
                continue

            line_idx = item["line_idx"]
            line_texts.setdefault(line_idx, []).append((item["box_idx"], text))

            original_box = [
                [int(pt[0] * sx_d2o), int(pt[1] * sy_d2o)]
                for pt in item["box"]
            ]
            details.append({"box": original_box, "text": text, "confidence": conf})

        full_text = ""
        for line_idx in sorted(line_texts.keys()):
            tokens = sorted(line_texts[line_idx], key=lambda x: x[0])
            line_str = " ".join(t for _, t in tokens)
            if line_str:
                full_text += line_str + "\n"

        logger.info("Kết quả: %d cụm chữ hợp lệ", len(details))

        return {
            "status":    "success",
            "text":      full_text.strip(),
            "details":   details,
            "imageSize": {"width": original_w, "height": original_h},
        }

    except Exception as e:
        logger.error(f"Lỗi: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def _extract_text_sync(contents: bytes) -> dict:
    """Decode bytes → numpy RGB, sau đó chạy pipeline chung."""
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        pil_img = ImageOps.exif_transpose(pil_img)
        img_np  = np.array(pil_img)
    except Exception as e:
        logger.error(f"Lỗi đọc ảnh: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

    return _extract_text_from_numpy(img_np)


# ─────────────────────────────────────────────────────────────────────────────
# PDF Processing — gọi thẳng _extract_text_from_numpy, không qua bytes
# ─────────────────────────────────────────────────────────────────────────────

_PDF_MAX_PAGES = int(os.environ.get("OCR_PDF_MAX_PAGES", "10"))
_PDF_DPI       = int(os.environ.get("OCR_PDF_DPI", "200"))

# Ma trận scale một lần, tránh tạo lại mỗi trang
_PDF_MAT = fitz.Matrix(_PDF_DPI / 72, _PDF_DPI / 72)


def _extract_text_from_pdf_sync(contents: bytes) -> dict:
    """
    Render từng trang PDF → numpy array → OCR.
    Khác bản cũ: bỏ bước PIL→PNG encode/decode thừa (~30–80ms/trang).
    """
    try:
        pdf = fitz.open(stream=contents, filetype="pdf")
        total_pages = min(len(pdf), _PDF_MAX_PAGES)
        if total_pages == 0:
            pdf.close()
            return {"status": "success", "text": "", "details": [], "pages": 0}

        all_text    = []
        all_details = []
        pdf_preview_base64 = None

        for page_num in range(total_pages):
            page = pdf[page_num]
            pix  = page.get_pixmap(matrix=_PDF_MAT, colorspace=fitz.csRGB, alpha=False)

            # ── THAY ĐỔI: numpy trực tiếp, không encode PNG ─────────────────
            # .copy() bắt buộc vì pix.samples là buffer tạm của fitz,
            # sẽ bị giải phóng khi pix ra khỏi scope.
            img_np = np.frombuffer(pix.samples, dtype=np.uint8) \
                       .reshape(pix.height, pix.width, 3) \
                       .copy()
            pix = None  # giải phóng sớm, tránh giữ buffer của fitz

            if page_num == 0:
                import base64
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                pdf_preview_base64 = base64.b64encode(buffer).decode('utf-8')
                pdf_image_size = {"width": img_np.shape[1], "height": img_np.shape[0]}

            page_result = _extract_text_from_numpy(img_np)

            if page_result.get("status") == "success":
                page_text = page_result.get("text", "").strip()
                if page_text:
                    all_text.append(f"--- Trang {page_num + 1} ---\n{page_text}")
                all_details.extend(page_result.get("details", []))

        pdf.close()

        return {
            "status":  "success",
            "text":    "\n\n".join(all_text),
            "details": all_details,
            "pages":   total_pages,
            "pdfPreviewBase64": pdf_preview_base64,
            "imageSize": pdf_image_size if pdf_image_size else {"width": 1000, "height": 1000},
        }
    except Exception as e:
        logger.error(f"Lỗi PDF: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post("/api/extract-text")
async def extract_text(file: UploadFile = File(...)):
    if "detector" not in models or "recognizer" not in models:
        return {"status": "error", "message": "Model chưa sẵn sàng."}
    try:
        contents     = await file.read()
        content_type = (file.content_type or "").lower()
        filename     = (file.filename    or "").lower()
        is_pdf       = "pdf" in content_type or filename.endswith(".pdf")

        if is_pdf:
            logger.info("Nhận file PDF: %s", file.filename)
            result = await asyncio.to_thread(_extract_text_from_pdf_sync, contents)
        else:
            async with ocr_semaphore:
                result = await asyncio.to_thread(_extract_text_sync, contents)
        return result
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
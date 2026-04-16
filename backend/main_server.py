import os
import logging
import io
import asyncio
import re
import numpy as np
import uvicorn
import cv2
import torch
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

# ── Tắt log thừa từ PaddlePaddle / PaddleX ─────────────────────────────────
os.environ["GLOG_minloglevel"]                   = "2"
os.environ["FLAGS_call_stack_level"]             = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["HF_ENDPOINT"]                        = "https://hf-mirror.com"

# Fix lỗi ConvertPirAttribute2RuntimeAttribute: tắt PIR executor mới + OneDNN
os.environ["FLAGS_use_mkldnn"]                   = "0"
os.environ["FLAGS_use_onednn"]                   = "0"
os.environ["FLAGS_enable_pir_api"]               = "0"
os.environ["FLAGS_new_executor_use_local_scope"]  = "0"

import warnings
warnings.filterwarnings("ignore")

# Monkey-patch để ép PaddleX tắt MKLDNN/OneDNN trên CPU
try:
    import paddlex.inference.utils.misc as px_misc
    px_misc.is_mkldnn_available = lambda: False
except ImportError:
    pass

from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCR_SERVER")

models    = {}
ocr_semaphore = asyncio.Semaphore(1)

_DET_LIMIT         = int(os.environ.get("OCR_DET_LIMIT",   "960"))
_RECOG_LIMIT       = int(os.environ.get("OCR_RECOG_LIMIT", "1280"))
_RECOGNITION_WORKERS = int(os.environ.get("OCR_RECOGNITION_WORKERS", "3"))


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: PaddleOCR 3.x (detection only) + VietOCR (recognition)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(">>> ĐANG KHỞI TẠO: PaddleOCR (detect) + VietOCR (recognize)... (Bản Tối Ưu Siêu Tốc)")
    try:
        # PaddleOCR 3.x — dùng để detect vị trí text box (rec_polys)
        models["detector"] = PaddleOCR(
            lang='vi',
            use_textline_orientation=False,
            text_det_limit_side_len=_DET_LIMIT,
            text_det_limit_type='max',
            text_recognition_batch_size=1,   
            text_rec_score_thresh=0.0,        
        )
        _warmup_detector(models["detector"])

        # VietOCR — nhận diện tiếng Việt chính xác
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained']        = True
        config['device']                   = 'cpu'
        
        # [QUAN TRỌNG NHẤT]: Tắt beamsearch giúp tăng tốc x5 lần mà độ chính xác giữ nguyên
        config['predictor']['beamsearch']  = False  
        
        models["recognizer"] = Predictor(config)
        _warmup_recognizer(models["recognizer"])

        logger.info(">>> KHỞI TẠO THÀNH CÔNG! (PaddleOCR detect + VietOCR vi siêu tốc)")
    except Exception as e:
        logger.error(f">>> LỖI KHỞI TẠO: {e}", exc_info=True)
    yield
    models.clear()


def _warmup_detector(ocr: PaddleOCR):
    try:
        dummy = np.ones((64, 256, 3), dtype=np.uint8) * 200
        list(ocr.predict(dummy))
        logger.info(">>> Detector warmup xong.")
    except Exception as ex:
        logger.warning(f">>> Detector warmup lỗi (bình thường): {ex}")


def _warmup_recognizer(rec: Predictor):
    try:
        dummy = Image.fromarray(np.ones((32, 128, 3), dtype=np.uint8) * 200)
        rec.predict(dummy, return_prob=True)
        logger.info(">>> Recognizer warmup xong.")
    except Exception as ex:
        logger.warning(f">>> Recognizer warmup lỗi: {ex}")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def resize_to_limit(img_np: np.ndarray, limit: int) -> np.ndarray:
    h, w = img_np.shape[:2]
    longest = max(h, w)
    if longest <= limit:
        return img_np
    scale = limit / longest
    interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=interp)


def enhance_image(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if gray.std() > 55.0:
        return img_np
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def sharpen_for_ocr(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() > 400:
        return img_np
    blur = cv2.GaussianBlur(img_np, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(img_np, 1.5, blur, -0.5, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Parse detection boxes từ PaddleOCR 3.x predict()
# ─────────────────────────────────────────────────────────────────────────────

def _parse_det_boxes(ocr_result_list) -> list[dict]:
    boxes = []
    for page in ocr_result_list:
        try:
            polys = getattr(page, "rec_polys", None)
            if polys is None and isinstance(page, dict):
                polys = page.get("rec_polys", [])
            if polys is None:
                continue
            for poly in polys:
                try:
                    pts = np.array(poly, dtype=np.float32)
                    boxes.append({
                        "box": [[float(p[0]), float(p[1])] for p in pts],
                    })
                except Exception:
                    pass
        except Exception:
            pass
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# NMS + Noise filter
# ─────────────────────────────────────────────────────────────────────────────

def _get_rect(box_pts) -> tuple[float, float, float, float]:
    pts = np.array(box_pts)
    return (
        float(pts[:, 0].min()), float(pts[:, 1].min()),
        float(pts[:, 0].max()), float(pts[:, 1].max()),
    )


def _nms_and_filter_boxes(box_data: list[dict], img_w: int, img_h: int) -> list[dict]:
    if not box_data:
        return []
    img_area = img_w * img_h
    for b in box_data:
        r = _get_rect(b["box"])
        b["_rect"] = r
        b["_area"] = (r[2] - r[0]) * (r[3] - r[1])

    min_abs = img_area * 0.0005
    box_data = [b for b in box_data if b["_area"] >= min_abs]
    if not box_data:
        return []

    median_area = np.median([b["_area"] for b in box_data])
    box_data = [b for b in box_data if b["_area"] >= median_area * 0.08]
    if not box_data:
        return []

    sorted_boxes = sorted(box_data, key=lambda b: b["_area"], reverse=True)
    rects  = np.array([b["_rect"] for b in sorted_boxes])
    areas  = np.array([b["_area"] for b in sorted_boxes])
    order  = np.arange(len(sorted_boxes))
    keep   = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1   = np.maximum(rects[i, 0], rects[order[1:], 0])
        yy1   = np.maximum(rects[i, 1], rects[order[1:], 1])
        xx2   = np.minimum(rects[i, 2], rects[order[1:], 2])
        yy2   = np.minimum(rects[i, 3], rects[order[1:], 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        ios   = inter / np.maximum(areas[order[1:]], 1.0)
        inds  = np.where((iou < 0.40) & (ios < 0.60))[0]
        order = order[inds + 1]

    return [sorted_boxes[idx] for idx in keep]


# ─────────────────────────────────────────────────────────────────────────────
# Smart crop cho VietOCR
# ─────────────────────────────────────────────────────────────────────────────

def _smart_crop(img_np: np.ndarray,
                x_min: float, y_min: float,
                x_max: float, y_max: float,
                box_h: float) -> list[Image.Image]:
    h, w = img_np.shape[:2]

    x1 = max(0, int(x_min - 12))
    y1 = max(0, int(y_min -  8))
    x2 = min(w, int(x_max + 12))
    y2 = min(h, int(y_max +  5))
    crops = [Image.fromarray(img_np[y1:y2, x1:x2])]

    if box_h > 40:
        mx = int((x_max - x_min) * 0.08)
        my = int(box_h * 0.10)
        x1t = max(0, int(x_min + mx));  y1t = max(0, int(y_min + my))
        x2t = min(w, int(x_max - mx));  y2t = min(h, int(y_max - my))
        if x2t > x1t + 5 and y2t > y1t + 5:
            crops.append(Image.fromarray(img_np[y1t:y2t, x1t:x2t]))

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
            if abs(yc - y_sums[j] / counts[j]) < (h_sums[j] / counts[j]) * 0.45:
                lines[j].append(b)
                y_sums[j] += yc; h_sums[j] += h; counts[j] += 1
                placed = True
                break
        if not placed:
            lines.append([b]); y_sums.append(yc); h_sums.append(h); counts.append(1)

    means = [y_sums[j] / counts[j] for j in range(len(lines))]
    return [lines[j] for j in np.argsort(means, kind='stable')]


# ─────────────────────────────────────────────────────────────────────────────
# VietOCR batch recognition
# ─────────────────────────────────────────────────────────────────────────────

_TORCH_TOTAL_THREADS = torch.get_num_threads()
_TORCH_PER_WORKER    = max(1, _TORCH_TOTAL_THREADS // max(1, _RECOGNITION_WORKERS))
_recognition_pool    = ThreadPoolExecutor(
    max_workers=_RECOGNITION_WORKERS,
    thread_name_prefix="vietocr",
)


def _batch_vietocr_predict(crops: list[Image.Image]) -> list[tuple[str, float]]:
    recognizer = models["recognizer"]
    if not crops:
        return []
    if len(crops) == 1:
        t, c = recognizer.predict(crops[0], return_prob=True)
        return [(t.strip(), float(c) if c else 1.0)]
    try:
        model     = recognizer.model
        transform = recognizer.transformers
        tensors   = [transform(c) for c in crops]
        max_w     = max(t.shape[2] for t in tensors)
        batch     = torch.zeros(len(tensors), 3, 32, max_w)
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


def _run_parallel_recognition(valid_crops: list[dict]) -> list[tuple[str, float]]:
    if not valid_crops:
        return []
    all_crops, item_ranges = [], []
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
        flat = [None] * len(chunks)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                flat[idx] = future.result()
            except Exception as e:
                logger.warning("Chunk %d failed: %s", idx, e)
                flat[idx] = [("", 0.0)] * len(chunks[idx])
        all_results = [r for chunk in flat for r in chunk]

    best = []
    for start, end in item_ranges:
        candidates = all_results[start:end]
        best.append(max(candidates, key=lambda x: x[1]))
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Text validation + post-processing
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_CHARS = frozenset(["1", "l", "I", "|", "!", "-", "'", "`"])

_VI_CHAR_RE = re.compile(r'[a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF.,;:!?()\[\]]')

_VI_OCR_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'  +'), ' '),                  
    (re.compile(r' +([.,;:!?])'), r'\1'),       
]


def _fix_text(text: str) -> str:
    for pat, rep in _VI_OCR_FIXES:
        text = pat.sub(rep, text)
    return text.strip()


def _is_valid_text(text: str, conf: float, box_w: int, box_h: int) -> bool:
    n = len(text)
    if n == 0:
        return False
    if n == 1 and text.isalpha() and text.isupper():
        return conf >= 0.65
    if n <= 2:
        if conf < 0.35:
            return False
        if not _VI_CHAR_RE.search(text):
            return False
    else:
        if conf < 0.15:
            return False
    if text in _NOISE_CHARS and box_h > 0 and (box_w / box_h) > 2.0:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main sync OCR function
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_sync(contents: bytes) -> dict:
    try:
        pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
        pil_img  = ImageOps.exif_transpose(pil_img)
        img_np   = np.array(pil_img)
        orig_h, orig_w = img_np.shape[:2]

        # ── Preprocess cho Detection ─────────────────────────────────────────
        det_img  = resize_to_limit(img_np, _DET_LIMIT)
        det_img  = enhance_image(det_img)
        det_img  = sharpen_for_ocr(det_img)
        det_h, det_w = det_img.shape[:2]

        # ── Preprocess cho Recognition (độ phân giải cao hơn) ───────────────
        recog_img = resize_to_limit(img_np, _RECOG_LIMIT)
        recog_img = enhance_image(recog_img)
        recog_h, recog_w = recog_img.shape[:2]

        sx_d2r = recog_w / det_w
        sy_d2r = recog_h / det_h
        sx_d2o = orig_w  / det_w
        sy_d2o = orig_h  / det_h

        # ── PHASE 1: PaddleOCR Detection ─────────────────────────────────────
        raw_output = list(models["detector"].predict(det_img))

        det_boxes = _parse_det_boxes(raw_output)
        if not det_boxes:
            return {"status": "success", "text": "", "details": [],
                    "imageSize": {"width": orig_w, "height": orig_h}}

        for item in det_boxes:
            pts = np.array(item["box"])
            y_min = float(pts[:, 1].min()); y_max = float(pts[:, 1].max())
            x_min = float(pts[:, 0].min()); x_max = float(pts[:, 0].max())
            item["y_center"] = (y_min + y_max) / 2
            item["x_min"]    = x_min
            item["y_min"]    = y_min
            item["height"]   = y_max - y_min
            item["width"]    = x_max - x_min

        # ── PHASE 2: NMS + noise filter ──────────────────────────────────────
        filtered = _nms_and_filter_boxes(det_boxes, det_w, det_h)
        if not filtered:
            return {"status": "success", "text": "", "details": [],
                    "imageSize": {"width": orig_w, "height": orig_h}}

        # ── PHASE 3: Line grouping ───────────────────────────────────────────
        lines = _group_boxes_into_lines(filtered)

        # ── PHASE 4: Smart crop → VietOCR ───────────────────────────────────
        valid_crops = []
        for line_idx, line in enumerate(lines):
            for box_idx, item in enumerate(sorted(line, key=lambda x: x["x_min"])):
                pts = np.array(item["box"], dtype=np.float32)
                x_min_d = max(0.0, pts[:, 0].min())
                y_min_d = max(0.0, pts[:, 1].min())
                x_max_d = min(float(det_w), pts[:, 0].max())
                y_max_d = min(float(det_h), pts[:, 1].max())
                bw = x_max_d - x_min_d
                bh = y_max_d - y_min_d
                if bh <= 0 or bw <= 0 or (bw / bh) < 1.2:
                    continue
                crops = _smart_crop(
                    recog_img,
                    x_min_d * sx_d2r, y_min_d * sy_d2r,
                    x_max_d * sx_d2r, y_max_d * sy_d2r,
                    bh * sy_d2r,
                )
                valid_crops.append({
                    "crops":    crops,
                    "box":      item["box"],
                    "line_idx": line_idx,
                    "box_idx":  box_idx,
                    "box_w":    int(bw),
                    "box_h":    int(bh),
                })

        if not valid_crops:
            return {"status": "success", "text": "", "details": [],
                    "imageSize": {"width": orig_w, "height": orig_h}}

        # ── PHASE 5: Parallel VietOCR recognition ────────────────────────────
        recog_results = _run_parallel_recognition(valid_crops)

        # ── PHASE 6: Assemble + post-process ─────────────────────────────────
        line_texts: dict[int, list[tuple[int, str]]] = {}
        details    = []

        for item, (raw_text, conf) in zip(valid_crops, recog_results):
            if not raw_text:
                continue
            text = _fix_text(raw_text)
            if not _is_valid_text(text, conf, item["box_w"], item["box_h"]):
                continue

            line_idx = item["line_idx"]
            line_texts.setdefault(line_idx, []).append((item["box_idx"], text))

            original_box = [
                [int(pt[0] * sx_d2o), int(pt[1] * sy_d2o)]
                for pt in item["box"]
            ]
            details.append({"box": original_box, "text": text, "confidence": round(conf, 4)})

        full_text = ""
        for line_idx in sorted(line_texts.keys()):
            tokens   = sorted(line_texts[line_idx], key=lambda x: x[0])
            line_str = " ".join(t for _, t in tokens)
            if line_str:
                full_text += line_str + "\n"

        logger.info("Kết quả: %d cụm chữ", len(details))
        return {
            "status":    "success",
            "text":      full_text.strip(),
            "details":   details,
            "imageSize": {"width": orig_w, "height": orig_h},
        }

    except Exception as e:
        logger.error(f"Lỗi xử lý ảnh: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/extract-text")
async def extract_text(image: UploadFile = File(...)):
    if "detector" not in models or "recognizer" not in models:
        return {"status": "error", "message": "Model chưa sẵn sàng."}
    try:
        contents = await image.read()
        async with ocr_semaphore:
            result = await asyncio.to_thread(_extract_text_sync, contents)
        return result
    except Exception as e:
        logger.error(f"Lỗi endpoint: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model":  "PaddleOCR-det + VietOCR-vgg_transformer (Fast Mode)",
        "lang":   "vi",
        "ready":  "detector" in models and "recognizer" in models,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
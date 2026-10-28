from __future__ import annotations

import logging
import json
import math
import threading
import warnings
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .inference_service import InferenceService
from .image_io import load_rgb_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
MAX_PREFERENCE_LENGTH = 2000
MAX_REGION_CONTROLS = 150
# A local demo shares one CNN and lazily loaded optional models across requests.
# Serialising work avoids duplicate model loads and concurrent full-image allocations.
SERVICE_LOCK = threading.RLock()

app = FastAPI(title="Preference Image Studio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
service = InferenceService()
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")


def _save_upload(upload: UploadFile, directory: Path, name: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, "Please upload a PNG, JPG/JPEG, or TIF/TIFF image.")
    path = directory / f"{name}{suffix}"
    total = 0
    with path.open("wb") as target:
        while chunk := upload.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Please upload an image smaller than 20 MiB.")
            target.write(chunk)
    if not total:
        raise HTTPException(400, "Please upload a non-empty image.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                if opened.format not in {"PNG", "JPEG", "TIFF"}:
                    raise HTTPException(400, "Please upload a PNG, JPG/JPEG, or TIF/TIFF image.")
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Please resize the image to at most 4 million pixels.")
                if getattr(opened, "n_frames", 1) != 1:
                    raise HTTPException(400, "Please upload a single-frame image.")
                opened.verify()
            # verify() does not necessarily decode the complete pixel payload.
            with Image.open(path) as opened:
                opened.load()
                if opened.mode == "F":
                    raise HTTPException(400, "Please use 8-bit RGB or 16-bit grayscale; floating-point images are unsupported.")
                if opened.mode == "I":
                    pixels = np.asarray(opened)
                    if pixels.min() < 0 or pixels.max() > 65535:
                        raise HTTPException(400, "Please use integer grayscale values within [0, 65535].")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise HTTPException(413, "Please resize the image to at most 4 million pixels.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise HTTPException(400, "Please upload a valid, readable image.") from error
    return path


def _regional_controls(raw: str) -> list[dict]:
    if len(raw) > 100_000:
        raise HTTPException(400, "Regional controls are too large.")
    try:
        regions = json.loads(raw)
    except (json.JSONDecodeError, ValueError, RecursionError) as error:
        raise HTTPException(400, "Regional controls must be valid JSON.") from error
    if not isinstance(regions, list) or len(regions) > MAX_REGION_CONTROLS:
        raise HTTPException(400, "Regional controls must be a list of at most 150 regions.")
    seen = set()
    normalised = []
    for region in regions:
        if not isinstance(region, dict) or set(region) - {"region_id", "r", "g", "b", "contrast", "strength"}:
            raise HTTPException(400, "Each region must contain a region_id and valid control fields.")
        region_id = region.get("region_id")
        if isinstance(region_id, str) and len(region_id) <= 8 and region_id.isascii() and region_id.isdecimal():
            region_id = int(region_id)
        if type(region_id) is not int or region_id < 0 or region_id in seen:
            raise HTTPException(400, "Region IDs must be unique non-negative integers.")
        seen.add(region_id)
        values = {key: region.get(key, 0) for key in ("r", "g", "b", "contrast")}
        strength = region.get("strength", 1)
        for key, value in {**values, "strength": strength}.items():
            bound = (0, 1) if key == "strength" else (-0.3, 0.3)
            if type(value) not in (int, float) or not bound[0] <= value <= bound[1] or not math.isfinite(value):
                raise HTTPException(400, "Regional controls must be finite numbers within their slider ranges.")
        normalised.append({"region_id": region_id, **values, "strength": strength})
    return normalised


def _output() -> tuple[Path, str]:
    name = f"enhanced-{uuid4().hex}.png"
    return OUTPUTS / name, f"/outputs/{name}"


def _run(callback):
    try:
        with SERVICE_LOCK:
            result, image_url = callback()
        result["enhanced_image_url"] = image_url
        result.pop("output_path", None)
        if "checkpoint" in result:
            result["checkpoint"] = Path(result["checkpoint"]).name
        return result
    except HTTPException:
        raise
    except ValueError as error:
        if str(error).startswith(("segmentation session", "region_id")):
            raise HTTPException(400, str(error)) from error
        logger.exception("Enhancement failed")
        raise HTTPException(500, "Enhancement failed. Please try again.") from error
    except Exception as error:
        logger.exception("Enhancement failed")
        raise HTTPException(500, "Enhancement failed. Please try again.") from error


@app.post("/api/preview")
def preview(image: UploadFile = File(...)):
    def work():
        with tempfile.TemporaryDirectory(prefix="pie-preview-") as folder:
            source = _save_upload(image, Path(folder), "source")
            name = f"preview-{uuid4().hex}.jpg"
            target = OUTPUTS / name
            rgb = load_rgb_image(source)
            dimensions = list(rgb.size)
            rgb.thumbnail((1600, 1600))
            rgb.save(target, quality=92)
            return ({"dimensions": dimensions}, f"/outputs/{name}")
    return _run(work)


@app.post("/api/segment")
def segment(image: UploadFile = File(...)):
    def work():
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="pie-segment-") as folder:
            source = _save_upload(image, Path(folder), "source")
            segment_id, label_map, _, cache_hit = service.segment_image(source)
            from .z3_use.z1_segmentation import IGNORE_LABEL_ID, LABELS
            regions = []
            for label_id in map(int, np.unique(label_map)):
                if label_id == IGNORE_LABEL_ID or label_id >= len(LABELS):
                    continue
                mask = label_map == label_id
                name = f"mask-{segment_id[:12]}-{label_id}.png"
                overlay = np.zeros((*label_map.shape, 4), dtype=np.uint8)
                overlay[mask] = [200, 240, 106, 58]
                Image.fromarray(overlay, "RGBA").save(OUTPUTS / name)
                regions.append({
                    "region_id": label_id,
                    "label": LABELS[label_id],
                    "coverage": float(mask.mean()),
                    "overlay_url": f"/outputs/{name}",
                })
            return ({
                "segment_id": segment_id,
                "regions": regions,
                "dimensions": [int(label_map.shape[1]), int(label_map.shape[0])],
                "cache_hit": cache_hit,
                "segmentation_seconds": time.perf_counter() - started,
            }, "")
    result = _run(work)
    result.pop("enhanced_image_url", None)
    return result


@app.get("/api/segment/{segment_id}/region")
def region_at(segment_id: str, x: int, y: int):
    try:
        with SERVICE_LOCK:
            return {"region": service.region_at(segment_id, x, y)}
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/health")
def health():
    return {
        "backend_ready": True,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cnn_ready": True,
        "segmentation": "ready" if service.adapter.segmentation_ready else "not-loaded",
        "qwen": "ready" if service.adapter.llm_ready else "not-loaded",
    }


@app.post("/api/enhance/text")
def enhance_text(image: UploadFile = File(...), preference: str = Form(...)):
    if not preference.strip():
        raise HTTPException(400, "Please describe your preference first.")
    if len(preference) > MAX_PREFERENCE_LENGTH:
        raise HTTPException(400, "Please keep your preference under 2000 characters.")
    def work():
        with tempfile.TemporaryDirectory(prefix="pie-text-") as folder:
            source = _save_upload(image, Path(folder), "source")
            output, url = _output()
            return service.enhance_text(source, preference, output), url
    return _run(work)


@app.post("/api/enhance/reference")
def enhance_reference(source: UploadFile = File(...), reference: UploadFile = File(...),
                      strength: float = Form(0.7), region_aware: bool = Form(True)):
    if not 0 <= strength <= 1:
        raise HTTPException(400, "Reference strength must be between 0 and 1.")

    def work():
        with tempfile.TemporaryDirectory(prefix="pie-reference-") as folder:
            folder = Path(folder)
            source_path = _save_upload(source, folder, "source")
            reference_path = _save_upload(reference, folder, "reference")
            output, url = _output()
            return service.enhance_reference(
                source_path, reference_path, strength, region_aware, output
            ), url
    return _run(work)


@app.post("/api/enhance/manual")
def enhance_manual(image: UploadFile = File(...), red: float = Form(0), green: float = Form(0),
                   blue: float = Form(0), contrast: float = Form(0),
                   strength: float = Form(1)):
    values = {"r": red, "g": green, "b": blue, "contrast": contrast}
    if any(not -0.3 <= value <= 0.3 for value in values.values()) or not 0 <= strength <= 1:
        raise HTTPException(400, "Controls must be within [-0.3, 0.3] and strength within [0, 1].")

    def work():
        with tempfile.TemporaryDirectory(prefix="pie-manual-") as folder:
            source = _save_upload(image, Path(folder), "source")
            output, url = _output()
            return service.enhance_manual(source, values, strength, output), url
    return _run(work)


@app.post("/api/enhance/manual-regions")
def enhance_manual_regions(
    image: UploadFile = File(...), segment_id: str = Form(...), regions_json: str = Form("[]"),
    global_red: float = Form(0), global_green: float = Form(0), global_blue: float = Form(0),
    global_contrast: float = Form(0), global_strength: float = Form(1),
):
    global_values = {
        "r": global_red, "g": global_green, "b": global_blue, "contrast": global_contrast,
    }
    regions = _regional_controls(regions_json)
    if any(not -0.3 <= value <= 0.3 for value in global_values.values()) or not 0 <= global_strength <= 1:
        raise HTTPException(400, "Invalid global controls.")

    def work():
        with tempfile.TemporaryDirectory(prefix="pie-manual-regions-") as folder:
            source = _save_upload(image, Path(folder), "source")
            output, url = _output()
            return service.enhance_manual_regions(
                source, segment_id, global_values, global_strength, regions, output
            ), url
    return _run(work)


@app.get("/")
def root():
    return {
        "name": "Preference Image Studio API",
        "status": "running",
        "frontend": "http://127.0.0.1:5173",
        "health": "/api/health",
        "docs": "/docs",
    }

from __future__ import annotations

import time
import hashlib
from threading import RLock
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .image_io import load_rgb_array
from .model.network import DEFAULT_CHECKPOINT, PreferenceControlledCNN
from .preference_adapter import PreferenceAdapter
from .reference_adapter import ReferenceAdapter


class InferenceService:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PreferenceControlledCNN(checkpoint_path).to(self.device).eval()
        self.adapter = PreferenceAdapter()
        self.reference_adapter = ReferenceAdapter()
        self._segment_cache = {}
        self._lock = RLock()

    @staticmethod
    def _digest(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def segment_image(self, image_path):
        with self._lock:
            segment_id = self._digest(image_path)
            cached = self._segment_cache.get(segment_id)
            if cached is not None:
                return segment_id, cached[0], cached[1], True
            label_map, regions = self.adapter.segment(image_path)
            self._segment_cache[segment_id] = (label_map, regions)
            while len(self._segment_cache) > 8:
                self._segment_cache.pop(next(iter(self._segment_cache)))
            return segment_id, label_map, regions, False

    def cached_segmentation(self, image_path, segment_id):
        with self._lock:
            if self._digest(image_path) != segment_id or segment_id not in self._segment_cache:
                raise ValueError("segmentation session does not match this source image; segment it again")
            return self._segment_cache[segment_id][0]

    def region_at(self, segment_id, x, y):
        with self._lock:
            if segment_id not in self._segment_cache:
                raise ValueError("segmentation session expired; segment the image again")
            label_map, _ = self._segment_cache[segment_id]
            if not 0 <= x < label_map.shape[1] or not 0 <= y < label_map.shape[0]:
                raise ValueError("coordinates are outside the source image")
            label_id = int(label_map[y, x])
            from .z3_use.z1_segmentation import IGNORE_LABEL_ID, LABELS
            if label_id == IGNORE_LABEL_ID or not 0 <= label_id < len(LABELS):
                return None
            return {"region_id": label_id, "label": LABELS[label_id]}

    @staticmethod
    def _load_image(path):
        return load_rgb_array(path)

    def _enhance(self, original, diffmap, output_path, details, started):
        with self._lock:
            output_path = Path(output_path)
            if (diffmap.shape != (4, original.shape[0], original.shape[1])
                    or not np.isfinite(diffmap).all() or np.abs(diffmap).max() > np.float32(0.3)):
                raise ValueError(f"invalid diffmap shape/values: {diffmap.shape}")
            image_tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            diffmap_tensor = torch.from_numpy(diffmap).unsqueeze(0).float().to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            cnn_started = time.perf_counter()
            with torch.inference_mode():
                combined_shape = tuple(torch.cat([image_tensor, diffmap_tensor], dim=1).shape)
                delta = self.model(image_tensor, diffmap_tensor)
                output = torch.clamp(image_tensor + delta, 0.0, 1.0)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            cnn_seconds = time.perf_counter() - cnn_started
            output_array = (output[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(output_array).save(output_path)
            return {
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
                "checkpoint": self.model.checkpoint_path.name,
                "checkpoint_loaded": True,
                "original_dimensions": [int(original.shape[1]), int(original.shape[0])],
                "image_tensor_shape": list(image_tensor.shape),
                "diffmap_tensor_shape": list(diffmap_tensor.shape),
                "diffmap_channel_min": [float(diffmap_tensor[:, index].min()) for index in range(4)],
                "diffmap_channel_max": [float(diffmap_tensor[:, index].max()) for index in range(4)],
                "combined_shape": list(combined_shape),
                "delta_shape": list(delta.shape),
                "output_shape": list(output.shape),
                "output_min": float(output.min()),
                "output_max": float(output.max()),
                "has_nan": bool(torch.isnan(output).any()),
                "has_inf": bool(torch.isinf(output).any()),
                "cnn_inference_seconds": cnn_seconds,
                "total_processing_seconds": time.perf_counter() - started,
                "output_path": str(output_path.resolve()),
                **details,
            }

    def enhance_text(self, image_path, preference, output_path):
        started = time.perf_counter()
        image_path = Path(image_path)
        original = self._load_image(image_path)
        segment_id, label_map, _, cache_hit = self.segment_image(image_path)
        diffmap, details = self.adapter.from_text(image_path, preference, label_map)
        details.update({"segment_id": segment_id, "segmentation_cache_hit": cache_hit})
        return self._enhance(original, diffmap, output_path, details, started)

    def enhance_manual(self, image_path, values, strength, output_path):
        started = time.perf_counter()
        original = self._load_image(image_path)
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("manual strength must be within [0, 1]")
        diffmap = self.adapter.from_manual(original.shape[0], original.shape[1], **values) * strength
        return self._enhance(original, diffmap, output_path, {
            "strength": strength,
            "conditioning": {key: float(value * strength) for key, value in values.items()},
        }, started)

    def enhance_manual_regions(self, image_path, segment_id, global_values, global_strength,
                               regions, output_path):
        started = time.perf_counter()
        original = self._load_image(image_path)
        label_map = self.cached_segmentation(image_path, segment_id)
        if label_map.shape != original.shape[:2]:
            raise ValueError("cached segmentation is not aligned with the source image")
        diffmap, applied = self.adapter.from_manual_regions(
            label_map, global_values, global_strength, regions
        )
        return self._enhance(original, diffmap, output_path, {
            "segment_id": segment_id,
            "global_conditioning": global_values,
            "global_strength": global_strength,
            "regional_conditioning": applied,
            "segmentation_cache_hit": True,
        }, started)

    def enhance_reference(self, source_path, reference_path, strength, region_aware, output_path):
        started = time.perf_counter()
        source, reference = self._load_image(source_path), self._load_image(reference_path)
        source_labels = reference_labels = None
        source_regions = reference_regions = []
        if region_aware:
            _, source_labels, source_regions, _ = self.segment_image(source_path)
            _, reference_labels, reference_regions, _ = self.segment_image(reference_path)
        from .z3_use.z1_segmentation import LABELS
        diffmap, details = self.reference_adapter.build(
            source, reference, strength, source_labels, reference_labels, LABELS
        )
        details.update({
            "source_dimensions": [source.shape[1], source.shape[0]],
            "reference_dimensions": [reference.shape[1], reference.shape[0]],
            "source_regions": source_regions,
            "reference_regions": reference_regions,
        })
        return self._enhance(source, diffmap, output_path, details, started)

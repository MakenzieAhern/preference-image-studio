from __future__ import annotations

from threading import RLock

import numpy as np
from PIL import Image, ImageFilter


class PreferenceAdapter:
    """The teammate z3_use text -> segmentation -> LLM -> four-channel diffmap flow."""

    def __init__(self):
        self._segmentation = None
        self._llm = None
        self._load_lock = RLock()

    def _load_once(self):
        with self._load_lock:
            if self._segmentation is None:
                from .z3_use.z1_segmentation import load_segmentation_model
                self._segmentation = load_segmentation_model()
            if self._llm is None:
                from .z3_use.z2_llm_instruction import load_llm
                self._llm = load_llm()

    @property
    def segmentation_ready(self):
        return self._segmentation is not None

    @property
    def llm_ready(self):
        return self._llm is not None

    def segment(self, image_path):
        with self._load_lock:
            if self._segmentation is None:
                from .z3_use.z1_segmentation import load_segmentation_model
                self._segmentation = load_segmentation_model()
            from .z3_use.z1_segmentation import present_labels, run_segmentation
            processor, segmenter = self._segmentation
            label_map = run_segmentation(str(image_path), processor, segmenter)
            return label_map, present_labels(label_map)

    def from_text(self, image_path, instruction, label_map=None):
        with self._load_lock:
            if not instruction or not instruction.strip():
                raise ValueError("preference text cannot be empty")
            self._load_once()
            from .z3_use.z1_segmentation import present_labels, run_segmentation
            from .z3_use.z2_llm_instruction import generate_region_json
            from .z3_use.z3_diffmap import build_diffmap_from_json, diffmap_to_chw

            processor, segmenter = self._segmentation
            tokenizer, llm = self._llm
            if label_map is None:
                label_map = run_segmentation(str(image_path), processor, segmenter)
            regions = present_labels(label_map)
            region_json = generate_region_json(instruction.strip(), regions, tokenizer, llm)
            diffmap = diffmap_to_chw(build_diffmap_from_json(label_map, region_json)).astype(np.float32)
            return diffmap, {"regions": regions, "region_json": region_json}

    @staticmethod
    def from_manual(height, width, r=0.0, g=0.0, b=0.0, contrast=0.0):
        values = np.asarray([r, g, b, contrast], dtype=np.float32)
        if not np.isfinite(values).all() or (np.abs(values) > 0.3).any():
            raise ValueError("manual r/g/b/contrast values must be finite and within [-0.3, 0.3]")
        return np.broadcast_to(values[:, None, None], (4, height, width)).copy()

    @staticmethod
    def from_manual_regions(label_map, global_values, global_strength, regions):
        if label_map.ndim != 2 or not np.issubdtype(label_map.dtype, np.integer):
            raise ValueError("label_map must be a two-dimensional integer array")
        if not np.isfinite(global_strength) or not 0 <= global_strength <= 1:
            raise ValueError("global strength must be within [0, 1]")
        if not isinstance(regions, list):
            raise ValueError("regional controls must be a list")
        height, width = label_map.shape
        diffmap = PreferenceAdapter.from_manual(height, width, **global_values) * global_strength
        applied = []
        seen = set()
        for region in regions:
            if not isinstance(region, dict) or "region_id" not in region:
                raise ValueError("each regional control must include region_id")
            raw_id = region["region_id"]
            try:
                region_id = int(raw_id)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("region_id must be an integer") from error
            if isinstance(raw_id, bool) or (not isinstance(raw_id, str) and raw_id != region_id) or region_id < 0:
                raise ValueError("region_id must be a non-negative integer")
            if region_id in seen:
                raise ValueError("regional controls cannot contain duplicate region IDs")
            seen.add(region_id)
            mask = label_map == region_id
            if not mask.any():
                raise ValueError(f"region_id {region_id} is not present in this image")
            values = {key: float(region.get(key, 0)) for key in ("r", "g", "b", "contrast")}
            adjustment = PreferenceAdapter.from_manual(height, width, **values)
            strength = float(region.get("strength", 1))
            if not 0 <= strength <= 1:
                raise ValueError("region strength must be within [0, 1]")
            soft_mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(radius=2)),
                dtype=np.float32,
            ) / 255.0
            diffmap += adjustment * soft_mask[None] * strength
            applied.append({"region_id": region_id, "strength": strength, **values})
        diffmap = np.clip(diffmap, -0.3, 0.3).astype(np.float32)
        return diffmap, applied

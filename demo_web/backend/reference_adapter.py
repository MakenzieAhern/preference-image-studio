from __future__ import annotations

import numpy as np


CHANNELS = ("red", "green", "blue", "contrast")


def _stats(image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    pixels = image.reshape(-1, 3) if mask is None else image[mask]
    if mask is not None and len(pixels) < 32:
        raise ValueError("region is too small for stable reference statistics")
    luminance = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    low, high = np.percentile(luminance, [5, 95])
    trimmed = pixels[(luminance >= low) & (luminance <= high)]
    if not len(trimmed):
        trimmed = pixels
    return {
        "rgb": np.median(trimmed, axis=0),
        "luminance": float(np.median(luminance)),
        "contrast": float(np.percentile(luminance, 90) - np.percentile(luminance, 10)),
    }


def _adjustment(source: np.ndarray, reference: np.ndarray,
                source_mask: np.ndarray | None = None,
                reference_mask: np.ndarray | None = None) -> np.ndarray:
    source_stats = _stats(source, source_mask)
    reference_stats = _stats(reference, reference_mask)
    rgb = reference_stats["rgb"] - source_stats["rgb"]
    contrast = reference_stats["contrast"] - source_stats["contrast"]
    return np.clip(np.append(rgb, contrast), -0.3, 0.3).astype(np.float32)


class ReferenceAdapter:
    """Deterministic image-statistics transfer into the CNN's real RGBC contract."""

    @staticmethod
    def build(source: np.ndarray, reference: np.ndarray, strength: float = 0.7,
              source_labels: np.ndarray | None = None,
              reference_labels: np.ndarray | None = None,
              label_names: list[str] | None = None) -> tuple[np.ndarray, dict]:
        for name, image in (("source", source), ("reference", reference)):
            if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
                raise ValueError(f"{name} must be a non-empty RGB image")
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f"{name} pixels must be finite and within [0, 1]")
        if (source_labels is None) != (reference_labels is None):
            raise ValueError("both source and reference labels are required for regional transfer")
        if source_labels is not None and (source_labels.shape != source.shape[:2] or reference_labels.shape != reference.shape[:2]):
            raise ValueError("semantic label maps must align with their source images")
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("reference strength must be within [0, 1]")
        global_values = _adjustment(source, reference)
        height, width = source.shape[:2]
        diffmap = np.broadcast_to(global_values[:, None, None], (4, height, width)).copy()
        matched = []

        if source_labels is not None and reference_labels is not None and label_names:
            shared = set(np.unique(source_labels)) & set(np.unique(reference_labels))
            for label_id in sorted(shared):
                label_id = int(label_id)
                if label_id < 0 or label_id >= len(label_names):
                    continue
                source_mask = source_labels == label_id
                reference_mask = reference_labels == label_id
                # Tiny masks are unstable and visually unimportant.
                if source_mask.mean() < 0.005 or reference_mask.mean() < 0.005:
                    continue
                try:
                    values = _adjustment(source, reference, source_mask, reference_mask)
                except ValueError:
                    continue
                diffmap[:, source_mask] = values[:, None]
                matched.append({
                    "name": label_names[label_id],
                    "adjustments": dict(zip(CHANNELS, map(float, values))),
                })

        diffmap *= np.float32(strength)
        if diffmap.shape != (4, height, width) or not np.isfinite(diffmap).all():
            raise ValueError("reference adapter produced an invalid conditioning map")
        return diffmap, {
            "mode_used": "region-aware" if matched else "global-fallback",
            "matched_regions": matched,
            "global_adjustments": dict(zip(CHANNELS, map(float, global_values))),
            "final_global_adjustments": dict(zip(CHANNELS, map(float, global_values * strength))),
            "strength": strength,
        }

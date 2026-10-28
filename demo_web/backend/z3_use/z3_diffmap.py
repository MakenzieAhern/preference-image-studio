"""Convert a semantic editing plan into bounded spatial R/G/B/contrast controls."""
import numpy as np
from .z1_segmentation import LABELS

LABEL_NAME_TO_ID = {name: idx for idx, name in enumerate(LABELS)}
CHANNELS = ("r", "g", "b", "contrast")


def _values(item):
    if not isinstance(item, dict):
        raise ValueError("each region must contain an adjustment object")
    try:
        values = np.array([float(item.get(key, 0)) for key in CHANNELS], dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("adjustment values must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError("adjustment values must be finite")
    return values


def build_diffmap_from_json(label_map, region_json):
    if label_map.ndim != 2 or not np.issubdtype(label_map.dtype, np.integer):
        raise ValueError("label_map must be a two-dimensional integer array")
    if not isinstance(region_json, dict):
        raise ValueError("the enhancement plan must be an object")
    height, width = label_map.shape
    global_values = _values(region_json.get("global", {}))
    diff_map = np.broadcast_to(global_values, (height, width, 4)).copy()
    for label_id in np.unique(label_map):
        if not 0 <= int(label_id) < len(LABELS):
            continue
        name = LABELS[int(label_id)]
        if name in region_json:
            diff_map[label_map == label_id] += _values(region_json[name])
    return np.clip(diff_map, -0.3, 0.3).astype(np.float32)


def diffmap_to_chw(diff_map):
    return diff_map.transpose(2, 0, 1)

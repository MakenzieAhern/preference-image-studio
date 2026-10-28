"""Lazy Mask2Former ADE20K segmentation; importing labels never downloads models."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..image_io import load_rgb_image

SEG_MODEL_REPO = "facebook/mask2former-swin-large-ade-semantic"
# A local directory override must contain the same ADE20K model and label mapping.
SEG_MODEL_PATH = os.environ.get("PIE_SEGMENTATION_MODEL_PATH", SEG_MODEL_REPO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
# Model predictions are 0..149. In particular, 127 is bicycle, not an ignore ID.
IGNORE_LABEL_ID = 255
LABELS = json.loads(Path(__file__).with_name("ade20k_labels.json").read_text(encoding="utf-8"))


def load_segmentation_model():
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    local_only = os.environ.get("PIE_MODEL_LOCAL_FILES_ONLY", "0") == "1"
    processor = AutoImageProcessor.from_pretrained(SEG_MODEL_PATH, local_files_only=local_only)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        SEG_MODEL_PATH, local_files_only=local_only
    ).to(DEVICE)
    labels = [model.config.id2label[index] for index in range(len(model.config.id2label))]
    if labels != LABELS:
        raise ValueError("Segmentation checkpoint must use the bundled ADE20K label mapping.")
    return processor, model.eval()


def load_rgb(image_path):
    return load_rgb_image(image_path)


def resize_for_processing(image, max_size):
    width, height = image.size
    scale = min(1.0, max_size / max(width, height))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(size, Image.Resampling.BILINEAR), image.size
    return image, image.size


@torch.no_grad()
def segment_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
    outputs = model(**inputs)
    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]
    return result.cpu().numpy().astype(np.uint8)


def run_segmentation(image_path, processor, model, label_save_path=None):
    image_full = load_rgb(image_path)
    image, original_size = resize_for_processing(image_full, IMAGE_SIZE)
    label_map = segment_image(image, processor, model)
    if original_size != image.size:
        label_map = np.asarray(Image.fromarray(label_map).resize(original_size, Image.Resampling.NEAREST))
    if label_save_path is not None:
        Image.fromarray(label_map).save(label_save_path)
    return label_map


def present_labels(label_map):
    return [LABELS[int(label)] for label in np.unique(label_map)
            if 0 <= int(label) < len(LABELS) and int(label) != IGNORE_LABEL_ID]

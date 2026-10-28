import numpy as np

from .reference_adapter import ReferenceAdapter


def main():
    rng = np.random.default_rng(7)
    source = rng.uniform(0.1, 0.7, (71, 53, 3)).astype(np.float32)
    warm_reference = np.clip(rng.uniform(0.2, 0.8, (37, 89, 3)) + [0.1, 0, -0.1], 0, 1).astype(np.float32)
    cool_reference = warm_reference[..., ::-1].copy()
    zero, zero_info = ReferenceAdapter.build(source, warm_reference, strength=0)
    full, full_info = ReferenceAdapter.build(source, warm_reference, strength=1)
    cool, _ = ReferenceAdapter.build(source, cool_reference, strength=1)
    assert zero.shape == full.shape == (4, 71, 53)
    assert np.allclose(zero, 0)
    assert np.allclose(full * 0.5, ReferenceAdapter.build(source, warm_reference, 0.5)[0])
    assert np.max(np.abs(full)) <= 0.3 and np.isfinite(full).all()
    assert not np.allclose(full, cool)
    assert zero_info["mode_used"] == full_info["mode_used"] == "global-fallback"

    source_labels = np.zeros((71, 53), dtype=np.uint8)
    reference_labels = np.zeros((37, 89), dtype=np.uint8)
    source_labels[:, :26] = 1
    reference_labels[:, :44] = 1
    regional, info = ReferenceAdapter.build(
        source, warm_reference, 0.7, source_labels, reference_labels, ["sky", "grass"]
    )
    assert regional.shape == (4, 71, 53) and info["mode_used"] == "region-aware"
    assert {item["name"] for item in info["matched_regions"]} == {"sky", "grass"}
    fallback, info = ReferenceAdapter.build(
        source, warm_reference, 1, np.zeros((71, 53), dtype=np.uint8),
        np.ones((37, 89), dtype=np.uint8), ["sky", "grass"]
    )
    assert fallback.shape == (4, 71, 53) and info["mode_used"] == "global-fallback"
    print("reference adapter checks passed")


if __name__ == "__main__":
    main()

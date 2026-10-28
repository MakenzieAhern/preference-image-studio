import numpy as np

from .preference_adapter import PreferenceAdapter


def main():
    labels = np.zeros((60, 90), dtype=np.uint8)
    labels[:, 30:60] = 1
    labels[:, 60:] = 2
    zeros = {"r": 0, "g": 0, "b": 0, "contrast": 0}

    global_map, _ = PreferenceAdapter.from_manual_regions(
        labels, {**zeros, "contrast": .03}, 1, []
    )
    assert global_map.shape == (4, 60, 90) and np.allclose(global_map[3], .03)

    one, _ = PreferenceAdapter.from_manual_regions(
        labels, zeros, 1, [{"region_id": 0, "r": .1, "strength": 1}]
    )
    assert one.shape == (4, 60, 90) and np.isfinite(one).all()
    assert one[0, 30, 10] > .099 and one[0, 30, 75] == 0
    assert 0 < one[0, 30, 30] < .1  # feathered boundary

    multiple, _ = PreferenceAdapter.from_manual_regions(labels, zeros, 1, [
        {"region_id": 0, "r": .3, "strength": 1},
        {"region_id": 1, "g": .2, "strength": .5},
        {"region_id": 2, "b": -.3, "contrast": .3, "strength": 1},
    ])
    assert not np.allclose(one, multiple)
    assert multiple.min() >= -.3 and multiple.max() <= .3

    clipped, _ = PreferenceAdapter.from_manual_regions(
        labels, {**zeros, "r": .25}, 1, [{"region_id": 0, "r": .2, "strength": 1}]
    )
    assert clipped.max() <= .3 and np.isclose(clipped[0, 30, 10], .3)
    print("region manual conditioning checks passed")


if __name__ == "__main__":
    main()

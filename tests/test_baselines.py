import numpy as np

from dermai.baselines import center_prior_heatmap
from dermai.localization import LocalizationEvaluator


def test_center_prior_peaks_at_center_and_tie_break_keeps_order():
    plain, broken = center_prior_heatmap(), center_prior_heatmap(tie_break=True)
    assert np.unravel_index(plain.argmax(), plain.shape) == (111, 111)
    assert np.unravel_index(broken.argmax(), broken.shape) == (111, 111)
    assert len(np.unique(broken)) == broken.size  # no ties left
    # Tie-break never reorders pixels that the plain heatmap already distinguishes.
    order = np.argsort(-broken.ravel(), kind="stable")
    assert np.all(np.diff(plain.ravel()[order]) <= 0)


def test_pointing_tolerance_zero_vs_fifteen():
    mask = np.zeros((224, 224), bool)
    mask[120:140, 120:140] = True
    heatmap = np.zeros((224, 224))
    heatmap[111, 111] = 1.0  # 9 px diagonal from the mask corner
    assert not LocalizationEvaluator(pointing_tolerance=0).compute_pointing_game(heatmap, mask)
    assert LocalizationEvaluator(pointing_tolerance=15).compute_pointing_game(heatmap, mask)

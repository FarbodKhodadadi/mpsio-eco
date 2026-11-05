import numpy as np
from mpsio_eco.statistics import paired_comparison


def test_paired_statistics_reproduce_values_reported_in_manuscript():
    proposed = np.array([0.8272, 0.8310, 0.8371, 0.8408, 0.8343])
    baseline = np.array([0.8265, 0.8165, 0.8233, 0.8223, 0.8193])
    result = paired_comparison(proposed, baseline)
    assert abs(result["mean_paired_difference"] - 0.0125) < 1e-10
    assert abs(result["paired_t_statistic"] - 4.086) < 0.002
    assert abs(result["paired_t_p_value"] - 0.0150) < 0.001
    assert abs(result["wilcoxon_p_value"] - 0.0625) < 1e-10

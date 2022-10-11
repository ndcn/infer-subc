from collections import defaultdict

DEFAULT_PARAMS = defaultdict(
    str,
    **{
        # "intensity_norm_param" : [0.5, 15]
        "intensity_norm_param": [0],
        "gaussian_smoothing_sigma": 1.34,
        "gaussian_smoothing_truncate_range": 3.0,
        "dot_2d_sigma": 2,
        "dot_2d_sigma_extra": 1,
        "dot_2d_cutoff": 0.025,
        "min_area": 10,
        "low_level_min_size": 100,
        "median_filter_size": 10,
    }
)

NAME = "infer_subc_2d"

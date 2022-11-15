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

NUC_CH = 0
LYSO_CH = 1
MITO_CH = 2
GOLGI_CH = 3
PEROXI_CH = 4
ER_CH = 5
LIPID_CH = 6
RESIDUAL_CH = 7

TEST_IMG_N = 5

ALL_CHANNELS = [NUC_CH,
                                 LYSO_CH,
                                MITO_CH,
                                GOLGI_CH,
                                PEROXI_CH,
                                ER_CH,
                                LIPID_CH,
                                RESIDUAL_CH]
import numpy as np
from typing import Tuple

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice


from infer_subc_2d.constants import (
    TEST_IMG_N,
    NUC_CH,
    LYSO_CH,
    MITO_CH,
    GOLGI_CH,
    PEROXI_CH,
    ER_CH,
    LIPID_CH,
    RESIDUAL_CH,
)

from infer_subc_2d.utils.img import (
    median_filter_slice_by_slice,
    min_max_intensity_normalization,
    apply_log_li_threshold,
    choose_agg_signal_zmax,
    select_z_from_raw,
)


def get_optimal_Z_image(img_data: np.ndarray) -> np.ndarray:
    """
    Procedure to infer _best_ Zslice from linearly unmixed input with fixed parameters
    """
    nuc_ch = NUC_CH
    ch_to_agg = (LYSO_CH, MITO_CH, GOLGI_CH, PEROXI_CH, ER_CH, LIPID_CH)
    optimal_Z = find_optimal_Z_params(img_data, nuc_ch, ch_to_agg)
    return select_z_from_raw(img_data, optimal_Z)


def find_optimal_Z(img_data: np.ndarray) -> int:
    """
    Procedure to infer _best_ Zslice from linearly unmixed input with fixed parameters
    """
    nuc_ch = NUC_CH
    ch_to_agg = (LYSO_CH, MITO_CH, GOLGI_CH, PEROXI_CH, ER_CH, LIPID_CH)
    return find_optimal_Z_params(img_data, nuc_ch, ch_to_agg)


def find_optimal_Z_params(raw_img: np.ndarray, nuc_ch: int, ch_to_agg: Tuple[int]) -> int:
    """
    Procedure to infer _best_ Zslice  from linearly unmixed input.

    Parameters:
    ------------
    raw_img: np.ndarray
        a ch,z,x,y - image containing florescent signal

    nuclei_ch: int
        holds the needed parameters

    nuclei_ch: int
        holds the needed parameters

    Returns:
    -------------
    opt_z:
        the "0ptimal" z-slice which has the most signal intensity for downstream 2D segmentation
    """

    # median filter in 2D / convert to float 0-1.   get rid of the "residual"

    struct_img = min_max_intensity_normalization(raw_img[nuc_ch].copy())
    med_filter_size = 4
    nuclei = median_filter_slice_by_slice(struct_img, size=med_filter_size)

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    nuclei = image_smoothing_gaussian_slice_by_slice(
        nuclei, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )
    threshold_factor = 0.9  # from cellProfiler
    thresh_min = 0.1
    thresh_max = 1.0
    struct_obj = apply_log_li_threshold(
        nuclei, threshold_factor=threshold_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )

    optimal_Z = choose_agg_signal_zmax(raw_img, ch_to_agg, mask=struct_obj)

    return optimal_Z

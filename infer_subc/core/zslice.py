import numpy as np
from typing import Tuple

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice


from infer_subc.constants import (
    TEST_IMG_N,
    NUC_CH,
    LYSO_CH,
    MITO_CH,
    GOLGI_CH,
    PEROX_CH,
    ER_CH,
    LD_CH,
    RESIDUAL_CH,
)

from infer_subc.core.img import (
    median_filter_slice_by_slice,
    min_max_intensity_normalization,
    apply_log_li_threshold,
    choose_agg_signal_zmax,
    select_z_from_raw,
)


def get_optimal_Z_image(img_data: np.ndarray, nuc_ch: int, ch_to_agg: Tuple[int]) -> np.ndarray:
    """
    Procedure to infer _best_ Zslice from linearly unmixed input

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    nuc_ch:
        channel with nuclei signal
    ch_to_agg:
        tuple of channels to aggregate for selecting Z

    Returns
    -------------
    image np.ndarray with single selected Z-slice   (Channels, 1, X, Y)

    """
    optimal_Z = find_optimal_Z(img_data, nuc_ch, ch_to_agg)
    return select_z_from_raw(img_data, optimal_Z)


def fixed_get_optimal_Z_image(img_data: np.ndarray) -> np.ndarray:
    """
    Procedure to infer _best_ Zslice from linearly unmixed input with fixed parameters
    """
    optimal_Z = fixed_find_optimal_Z(img_data)
    return select_z_from_raw(img_data, optimal_Z)


def fixed_find_optimal_Z(img_data: np.ndarray) -> int:
    """
    Procedure to infer _best_ Zslice from linearly unmixed input with fixed parameters
    """
    nuc_ch = NUC_CH
    ch_to_agg = (LYSO_CH, MITO_CH, GOLGI_CH, PEROX_CH, ER_CH, LD_CH)
    return find_optimal_Z(img_data, nuc_ch, ch_to_agg)


def find_optimal_Z(raw_img: np.ndarray, nuc_ch: int, ch_to_agg: Tuple[int]) -> int:
    """
    Procedure to infer _best_ Zslice  from linearly unmixed input.

    Parameters
    ------------
    raw_img:
        a ch,z,x,y - image containing florescent signal

    nuc_ch:
        channel with nuclei signal

    ch_to_agg:
        tuple of channels to aggregate for selecting Z

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
    thresh_factor = 0.9  # from cellProfiler
    thresh_min = 0.1
    thresh_max = 1.0
    struct_obj = apply_log_li_threshold(
        nuclei, thresh_factor=thresh_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )

    optimal_Z = choose_agg_signal_zmax(raw_img, ch_to_agg, mask=struct_obj)
    print(f"choosing _optimal_ z-slice::: {optimal_Z}")
    return optimal_Z

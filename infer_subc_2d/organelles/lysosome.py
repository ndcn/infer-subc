import numpy as np
from typing import Optional

from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice

from infer_subc_2d.constants import LYSO_CH

from infer_subc_2d.utils.img import (
    apply_mask,
    median_filter_slice_by_slice,
    min_max_intensity_normalization,
    size_filter_linear_size,
    select_channel_from_raw,
    filament_filter,
)

##########################
#  infer_LYSOSOMES
##########################
def infer_lysosome(
    in_img: np.ndarray,
    cytosol_mask: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    dot_scale_1: float,
    dot_cut_1: float,
    dot_scale_2: float,
    dot_cut_2: float,
    dot_scale_3: float,
    dot_cut_3: float,
    filament_scale: float,
    filament_cut: float,
    min_hole_w: int,
    max_hole_w: int,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer lysosome from linearly unmixed input

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    cytosol_mask:
        mask
    median_sz:
        width of median filter for signal
    gauss_sig:
        sigma for gaussian smoothing of  signal
    dot_scale:
        scales (log_sigma) for dot filter (1,2, and 3)
    dot_cut:
        threshold for dot filter thresholds (1,2,and 3)
    filament_scale:
        scale (log_sigma) for filament filter
    filament_cut:
        threshold for filament fitered threshold
    min_hole_w:
        hole filling min for nuclei post-processing
    max_hole_w:
        hole filling cutoff for nuclei post-processing
    small_obj_w:
        minimu object size cutoff for nuclei post-processing

    Returns
    -------------
    lysosome_object:
        mask defined extent of lysosome object

    """
    lyso_ch = LYSO_CH
    ###################
    # EXTRACT
    ###################
    lyso = select_channel_from_raw(in_img, lyso_ch)

    ###################
    # PRE_PROCESSING
    ###################
    lyso = min_max_intensity_normalization(lyso)

    lyso = median_filter_slice_by_slice(lyso, size=median_sz)

    lyso = image_smoothing_gaussian_slice_by_slice(lyso, sigma=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    # s2_param = [[5,0.09], [2.5,0.07], [1,0.01]]
    s2_param = [[dot_scale_1, dot_cut_1], [dot_scale_2, dot_cut_2], [dot_scale_3, dot_cut_3]]
    bw_spot = dot_2d_slice_by_slice_wrapper(lyso, s2_param)

    # f2_param = [[filament_scale, filament_cut]]
    # # f2_param = [[1, 0.15]]  # [scale_1, cutoff_1]
    # bw_filament = filament_2d_wrapper(lyso, f2_param)
    bw_filament = filament_filter(lyso, filament_scale, filament_cut)

    bw = np.logical_or(bw_spot, bw_filament)

    ###################
    # POST_PROCESSING
    ###################

    struct_obj = hole_filling(bw, hole_min=min_hole_w**2, hole_max=max_hole_w**2, fill_2d=True)

    struct_obj = apply_mask(struct_obj, cytosol_mask)

    struct_obj = size_filter_linear_size(struct_obj, min_size=small_obj_w**2, connectivity=1)

    return struct_obj


##########################
#  fixed_infer_nuclei
##########################
def fixed_infer_lysosome(in_img: np.ndarray, cytosol_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Procedure to infer lysosome from linearly unmixed input

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    cytosol_mask:
        mask

    Returns
    -------------
    nuclei_object:
        mask defined extent of NU
    """
    median_sz = 4
    gauss_sig = 1.34
    dot_scale_1 = 5
    dot_cut_1 = 0.09
    dot_scale_2 = 2.5
    dot_cut_2 = 0.07
    dot_scale_3 = 1
    dot_cut_3 = 0.01
    filament_scale = 1
    filament_cut = 0.15
    min_hole_w = 0
    max_hole_w = 25
    small_obj_w = 3

    return infer_lysosome(
        in_img,
        cytosol_mask,
        median_sz,
        gauss_sig,
        dot_cut_1,
        dot_scale_1,
        dot_cut_2,
        dot_scale_2,
        dot_cut_3,
        dot_scale_3,
        filament_scale,
        filament_cut,
        min_hole_w,
        max_hole_w,
        small_obj_w,
    )


def lysosome_spot_filter(in_img: np.ndarray) -> np.ndarray:
    """spot filter helper function for lysosome"""
    dot_scale_1 = 5
    dot_cut_1 = 0.09
    dot_scale_2 = 2.5
    dot_cut_2 = 0.07
    dot_scale_3 = 1
    dot_cut_3 = 0.01
    s2_param = [[dot_scale_1, dot_cut_1], [dot_scale_2, dot_cut_2], [dot_scale_3, dot_cut_3]]
    return dot_2d_slice_by_slice_wrapper(in_img, s2_param)


def lysosome_filiment_filter(in_img: np.ndarray) -> np.ndarray:
    """spot filter helper function for lysosome (DEPRICATED)"""
    filament_scale = 1
    filament_cut = 0.15
    f2_param = [[filament_scale, filament_cut]]
    # f2_param = [[1, 0.15]]  # [scale_1, cutoff_1]
    return filament_2d_wrapper(in_img, f2_param)

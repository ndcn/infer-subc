import numpy as np

from infer_subc_2d.constants import MITO_CH

from infer_subc_2d.utils.img import (
    size_filter_linear_size,
    size_filter_linear_size,
    vesselness_slice_by_slice,
    select_channel_from_raw,
    scale_and_smooth,
)

##########################
#  infer_mitochondria
##########################
def infer_mitochondria(
    in_img: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    vesselness_scale: float,
    vesselness_cut: float,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer mitochondria from linearly unmixed input.

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    median_sz:
        width of median filter for signal
    gauss_sig:
        sigma for gaussian smoothing of  signal
    vesselness_scale:
        scale (log_sigma) for vesselness filter
    vesselness_cut:
        threshold for vesselness fitered threshold
    small_obj_w:
        minimu object size cutoff for nuclei post-processing

    Returns
    -------------
    mitochondria_object
        mask defined extent of mitochondria
    """
    mito_ch = MITO_CH
    ###################
    # EXTRACT
    ###################
    mito = select_channel_from_raw(in_img, MITO_CH)

    ###################
    # PRE_PROCESSING
    ###################
    struct_img = scale_and_smooth(mito, median_sz=median_sz, gauss_sig=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    struct_img = vesselness_slice_by_slice(struct_img, sigma=vesselness_scale, cutoff=vesselness_cut, tau=0.75)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = size_filter_linear_size(struct_img, min_size=small_obj_w)

    return struct_obj


##########################
#  fixed_infer_mitochondria
##########################
def fixed_infer_mitochondria(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer mitochondria from linearly unmixed input

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    Returns
    -------------
    mitochondria_object
        mask defined extent of mitochondria
    """
    median_sz = 3
    gauss_sig = 1.4
    vesselness_scale = 1.5
    vesselness_cut = 0.05
    small_obj_w = 3

    return infer_mitochondria(in_img, median_sz, gauss_sig, vesselness_scale, vesselness_cut, small_obj_w)

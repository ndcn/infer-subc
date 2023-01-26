import numpy as np
from typing import Optional

from aicssegmentation.core.pre_processing_utils import (
    image_smoothing_gaussian_slice_by_slice,
)

from infer_subc_2d.constants import MITO_CH

from infer_subc_2d.utils.img import (
    apply_mask,
    median_filter_slice_by_slice,
    min_max_intensity_normalization,
    size_filter_2D,
    vesselness_slice_by_slice,
    select_channel_from_raw,
)

##########################
#  infer_mitochondria
##########################
def infer_mitochondria(
    in_img: np.ndarray,
    cytosol_mask: np.ndarray,
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
    soma_mask:
        mask default-None
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
    # PRE_PROCESSING
    ###################
    mito = select_channel_from_raw(in_img, MITO_CH)

    mito = min_max_intensity_normalization(mito)

    mito = median_filter_slice_by_slice(mito, size=median_sz)

    struct_img = image_smoothing_gaussian_slice_by_slice(mito, sigma=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    struct_img = vesselness_slice_by_slice(struct_img, sigma=vesselness_scale, cutoff=vesselness_cut, tau=0.75)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = apply_mask(struct_img, cytosol_mask)

    struct_obj = size_filter_2D(struct_obj, min_size=small_obj_w**2, connectivity=1)

    return struct_obj


##########################
#  fixed_infer_mitochondria
##########################
def fixed_infer_mitochondria(in_img: np.ndarray, cytosol_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Procedure to infer mitochondria from linearly unmixed input

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    cytosol_mask:
        mask

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

    return infer_mitochondria(in_img, cytosol_mask, median_sz, gauss_sig, vesselness_scale, vesselness_cut, small_obj_w)

import numpy as np
from typing import Optional

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.utils import hole_filling

from infer_subc_2d.utils.img import (
    apply_threshold,
    apply_mask,
    min_max_intensity_normalization,
    median_filter_slice_by_slice,
    size_filter_2D,
    select_channel_from_raw,
)
from infer_subc_2d.constants import LIPID_CH


##########################
#  infer_lipid
##########################
def infer_lipid(
    in_img: np.ndarray,
    cytosol_mask: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    method: str,
    thresh_factor: float,
    thresh_min: float,
    thresh_max: float,
    max_hole_w: int,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer peroxisome from linearly unmixed input.

    Parameters:
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels
    cytosol_mask: np.ndarray
        mask of cytosol
    median_sz: int
        width of median filter for signal
    gauss_sig: float
        sigma for gaussian smoothing of  signal
    method: str
        method for applying threshold.  "otsu"  or "li", "triangle", "median", "ave", "sauvola","multi_otsu","muiltiotsu"
    thresh_factor:float=1.0
        scaling value for threshold
    thresh_min= None or min
        absolute minumum for threshold
    thresh_max = None or max
        absolute maximum for threshold
    max_hole_w: int
        hole filling cutoff for nuclei post-processing
    small_obj_w: int
        minimu object size cutoff for nuclei post-processing
    Returns:
    -------------
    peroxi_object
        mask defined extent of peroxisome object
    """
    lipid_ch = LIPID_CH
    ###################
    # EXTRACT
    ###################
    lipid = select_channel_from_raw(in_img, lipid_ch)
    ###################
    # PRE_PROCESSING
    ###################
    lipid = min_max_intensity_normalization(lipid)

    lipid = median_filter_slice_by_slice(lipid, size=median_sz)

    lipid = image_smoothing_gaussian_slice_by_slice(lipid, sigma=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    bw = apply_threshold(
        lipid, method=method, thresh_factor=thresh_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )

    ###################
    # POST_PROCESSING
    ###################
    min_hole_w = 0
    struct_obj = hole_filling(bw, hole_min=min_hole_w**2, hole_max=max_hole_w**2, fill_2d=True)

    struct_obj = apply_mask(struct_obj, cytosol_mask)

    struct_obj = size_filter_2D(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_obj_w**2,
        connectivity=1,
    )

    return struct_obj


##########################
#  fixed_infer_lipid
##########################
def fixed_infer_lipid(in_img: np.ndarray, cytosol_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Procedure to infer soma from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters:
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels
     cytosol_mask: Optional[np.ndarray] = None
         mask

    Returns:
    -------------
    lipid_body_object
        mask defined extent of liipid body

    """

    median_sz = 2
    gauss_sig = 1.34
    method = "otsu"
    threshold_factor = 0.99  # from cellProfiler
    thresh_min = 0.5
    thresh_max = 1.0
    max_hole_w = 2.5
    small_obj_w = 4

    return infer_lipid(
        in_img,
        cytosol_mask,
        median_sz,
        gauss_sig,
        method,
        threshold_factor,
        thresh_min,
        thresh_max,
        max_hole_w,
        small_obj_w,
    )

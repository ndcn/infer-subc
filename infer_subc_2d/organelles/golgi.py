import numpy as np
from typing import Optional

from aicssegmentation.core.seg_dot import dot_3d_wrapper, dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.utils import topology_preserving_thinning
from aicssegmentation.core.MO_threshold import MO

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice

from infer_subc_2d.constants import GOLGI_CH

from infer_subc_2d.utils.img import (
    apply_mask,
    median_filter_slice_by_slice,
    min_max_intensity_normalization,
    size_filter_2D,
    select_channel_from_raw,
)

##########################
#  infer_golgi
##########################
def infer_golgi(
    in_img: np.ndarray,
    cytosol_mask: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    obj_min_area: int,
    thresh_method: str,
    min_thickness: int,
    thin: int,
    dot_scale: float,
    dot_cut: float,
    small_obj_w: int,
) -> np.ndarray:
    """
     Procedure to infer golgi from linearly unmixed input.

    Parameters:
     ------------
     in_img: np.ndarray
         a 3d image containing all the channels
    cytosol_mask: np.ndarray
        mask
     median_sz: int
         width of median filter for signal
     gauss_sig: float
         sigma for gaussian smoothing of  signal
     obj_min_area: int
         the size filter for excluding small object before applying local threshold
     thresh_method: str
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
     min_thinkness: int
         Half of the minimum width you want to keep from being thinned.
         For example, when the object width is smaller than 4, you don't
         want to make this part even thinner (may break the thin object
         and alter the topology), you can set this value as 2.
     thin: int
         the amount to thin (has to be an positive integer). The number of
          pixels to be removed from outter boundary towards center.
     dot_scale: float
         scales (log_sigma) for dot filter (1,2, and 3)
     dot_cut: float
         threshold for dot filter thresholds (1,2,and 3)
     small_obj_w: int
         minimu object size cutoff for nuclei post-processing

     Returns:
     -------------
    golgi_object
         mask defined extent of golgi object
    """
    golgi_ch = GOLGI_CH

    ###################
    # EXTRACT
    ###################
    golgi = select_channel_from_raw(in_img, golgi_ch)

    ###################
    # PRE_PROCESSING
    ###################
    golgi = min_max_intensity_normalization(golgi)

    golgi = median_filter_slice_by_slice(golgi, size=median_sz)

    golgi = image_smoothing_gaussian_slice_by_slice(golgi, sigma=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    bw = MO(golgi, global_thresh_method=thresh_method, object_minArea=obj_min_area)

    bw_thin = topology_preserving_thinning(bw, min_thickness, thin)

    s3_param = [(dot_cut, dot_scale)]
    bw_extra = dot_2d_slice_by_slice_wrapper(golgi, s3_param)
    # bw_extra = dot_3d_wrapper(golgi, s3_param)

    bw = np.logical_or(bw_extra, bw_thin)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = apply_mask(bw, cytosol_mask)

    struct_obj = size_filter_2D(struct_obj, min_size=small_obj_w**2, connectivity=1)

    return struct_obj


##########################
#  fixed_infer_golgi
##########################
def fixed_infer_golgi(in_img: np.ndarray, cytosol_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
     Procedure to infer golgi from linearly unmixed input with fixed parameters.

    Parameters:
     ------------
     in_img: np.ndarray
         a 3d image containing all the channels
     soma_mask: Optional[np.ndarray] = None
         mask

     Returns:
     -------------
    golgi_object
         mask defined extent of golgi object
    """

    median_sz = 4
    gauss_sig = 1.34
    obj_min_area = 1200
    thresh_method = "tri"
    min_thickness = 1.6
    thin = 1
    dot_scale = 1.6
    dot_cut = 0.02
    small_obj_w = 3

    return infer_golgi(
        in_img,
        cytosol_mask,
        median_sz,
        gauss_sig,
        obj_min_area,
        thresh_method,
        min_thickness,
        thin,
        dot_scale,
        dot_cut,
        small_obj_w,
    )

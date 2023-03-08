import numpy as np
from typing import Optional

from aicssegmentation.core.seg_dot import dot_3d_wrapper, dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.utils import topology_preserving_thinning

from infer_subc_2d.constants import GOLGI_CH

from infer_subc_2d.utils.img import (
    size_filter_linear_size,
    select_channel_from_raw,
    masked_object_thresh,
    scale_and_smooth,
)

##########################
#  infer_golgi
##########################
def infer_golgi(
    in_img: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    mo_method: str,
    mo_adjust: float,
    mo_cutoff_size: int,
    min_thickness: int,
    thin: int,
    dot_scale: float,
    dot_cut: float,
    small_obj_w: int,
) -> np.ndarray:

    """
     Procedure to infer golgi from linearly unmixed input.

    Parameters
     ------------
     in_img:
         a 3d image containing all the channels
     median_sz:
         width of median filter for signal
     mo_method:
          which method to use for calculating global threshold. Options include:
          "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
          "ave" refers the average of "triangle" threshold and "mean" threshold.
     mo_adjust:
         Masked Object threshold `local_adjust`
     mo_cutoff_size:
         Masked Object threshold `size_min`
     min_thinkness:
         Half of the minimum width you want to keep from being thinned.
         For example, when the object width is smaller than 4, you don't
         want to make this part even thinner (may break the thin object
         and alter the topology), you can set this value as 2.
     thin:
         the amount to thin (has to be an positive integer). The number of
          pixels to be removed from outter boundary towards center.
     dot_scale:
         scales (log_sigma) for dot filter (1,2, and 3)
     dot_cut:
         threshold for dot filter thresholds (1,2,and 3)
     small_obj_w:
         minimu object size cutoff for nuclei post-processing

     Returns
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
    golgi = scale_and_smooth(golgi, median_sz=median_sz, gauss_sig=gauss_sig)
    ###################
    # CORE_PROCESSING
    ###################
    # bw = MO(golgi, global_thresh_method=thresh_method, object_minArea=obj_min_area)
    bw = masked_object_thresh(golgi, th_method=mo_method, cutoff_size=mo_cutoff_size, th_adjust=mo_adjust)

    bw_thin = topology_preserving_thinning(bw, min_thickness, thin)

    s3_param = [(dot_cut, dot_scale)]
    bw_extra = dot_2d_slice_by_slice_wrapper(golgi, s3_param)
    # bw_extra = dot_3d_wrapper(golgi, s3_param)

    bw = np.logical_or(bw_extra, bw_thin)
    ###################
    # POST_PROCESSING
    ###################
    struct_obj = size_filter_linear_size(bw, min_size=small_obj_w, connectivity=1)

    return struct_obj


##########################
#  fixed_infer_golgi
##########################
def fixed_infer_golgi(in_img: np.ndarray, cytosol_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
     Procedure to infer golgi from linearly unmixed input.

     Parameters
     ------------
     in_img:
         a 3d image containing all the channels
     Returns
     -------------
    golgi_object
         mask defined extent of golgi object
    """

    median_sz = 4
    gauss_sig = 1.34
    mo_method = "tri"
    mo_adjust = 0.90
    mo_cutoff_size = 1200
    min_thickness = 1.6
    thin = 1
    dot_scale = 1.6
    dot_cut = 0.02
    small_obj_w = 3

    return infer_golgi(
        in_img,
        median_sz,
        gauss_sig,
        mo_method,
        mo_adjust,
        mo_cutoff_size,
        min_thickness,
        thin,
        dot_scale,
        dot_cut,
        small_obj_w,
    )

import numpy as np
from typing import Optional

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper

from infer_subc_2d.constants import PEROXI_CH

from infer_subc_2d.utils.img import (
    apply_mask,
    min_max_intensity_normalization,
    size_filter_linear_size,
    select_channel_from_raw,
)

##########################
#  infer_peroxisome
##########################
def infer_peroxisome(
    in_img: np.ndarray,
    cytosol_mask: np.ndarray,
    gauss_sig: float,
    dot_scale: float,
    dot_cut: float,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer peroxisome from linearly unmixed input.

    Parameters
     ------------
     in_img:
         a 3d image containing all the channels
     cytosol_mask:
         mask
     gauss_sig:
         sigma for gaussian smoothing of  signal
     dot_scale:
         scales (log_sigma) for dot filter (1,2, and 3)
     dot_cut:
         threshold for dot filter thresholds (1,2,and 3)
     small_obj_w:
         minimu object size cutoff for nuclei post-processing

     Returns
     -------------
     peroxi_object
         mask defined extent of peroxisome object
    """
    peroxi_ch = PEROXI_CH
    ###################
    # EXTRACT
    ###################
    peroxi = select_channel_from_raw(in_img, peroxi_ch)

    ###################
    # PRE_PROCESSING
    ###################
    peroxi = min_max_intensity_normalization(peroxi)

    peroxi = image_smoothing_gaussian_slice_by_slice(peroxi, sigma=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    s3_param = [[dot_scale, dot_cut]]
    bw = dot_2d_slice_by_slice_wrapper(peroxi, s3_param)

    ###################
    # POST_PROCESSING
    ###################
    # do_watershed = False
    # if do_watershed: # BUG: this makes bw into labels... but we are returning a binary object...
    #     minArea = 4
    #     mask_ = remove_small_objects(bw>0, min_size=minArea, connectivity=1, in_place=False)
    #     seed_ = dilation(peak_local_max(struct_img,labels=label(mask_), min_distance=2, indices=False), selem=ball(1))
    #     watershed_map = -1*ndi.distance_transform_edt(bw)
    #     bw = watershed(watershed_map, label(seed_), mask=mask_, watershed_line=True)
    struct_obj = apply_mask(bw, cytosol_mask)

    struct_obj = size_filter_linear_size(struct_obj, min_size=small_obj_w**2, connectivity=1)

    return struct_obj


##########################
#  fixed_infer_peroxisome
##########################
def fixed_infer_peroxisome(in_img: np.ndarray, cytosol_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
      Procedure to infer peroxisome from linearly unmixed input with fixed parameters.

    Parameters
     ------------
     in_img: np.ndarray
         a 3d image containing all the channels
     cytosol_mask:
         mask - default=None

     Returns
     -------------
     peroxi_object
         mask defined extent of peroxisome object
    """
    gauss_sig = 3.0
    dot_scale = 1.0
    dot_cut = 0.01
    small_obj_w = 2

    return infer_peroxisome(
        in_img,
        cytosol_mask,
        gauss_sig,
        dot_scale,
        dot_cut,
        small_obj_w,
    )

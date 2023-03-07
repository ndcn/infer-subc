import numpy as np
from typing import Optional

from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import edge_preserving_smoothing_3d

from infer_subc_2d.constants import ER_CH

from infer_subc_2d.utils.img import (
    apply_mask,
    min_max_intensity_normalization,
    size_filter_linear_size,
    select_channel_from_raw,
    filament_filter,
)

##########################
#  infer_endoplasmic_reticulum
##########################
def infer_endoplasmic_reticulum(
    in_img: np.ndarray,
    cytosol_mask: np.ndarray,
    filament_scale: float,
    filament_cut: float,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer peroxisome from linearly unmixed input.

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    cytosol_mask:
        mask of cytosol
    filament_scale:
        scale (log_sigma) for filament filter
    filament_cut:
        threshold for filament fitered threshold
    small_obj_w:
        minimu object size cutoff for nuclei post-processing
    Returns
    -------------
    peroxi_object
        mask defined extent of peroxisome object
    """
    er_ch = ER_CH
    ###################
    # EXTRACT
    ###################
    er = select_channel_from_raw(in_img, er_ch)

    ###################
    # PRE_PROCESSING
    ###################
    er = min_max_intensity_normalization(er)

    # edge-preserving smoothing (Option 2, used for Sec61B)
    er = edge_preserving_smoothing_3d(er)

    ###################
    # CORE_PROCESSING
    ###################
    # f2_param = [[filament_scale, filament_cut]]
    # struct_obj = filament_2d_wrapper(er, f2_param)
    struct_obj = filament_filter(er, filament_scale, filament_cut)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = apply_mask(struct_obj, cytosol_mask)

    struct_obj = size_filter_linear_size(struct_obj, min_size=small_obj_w**2, connectivity=1)

    return struct_obj


##########################
#  fixed_infer_endoplasmic_reticulum
##########################
def fixed_infer_endoplasmic_reticulum(in_img: np.ndarray, cytosol_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Procedure to infer endoplasmic rediculum from linearly unmixed input with *fixed parameters*

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    cytosol_mask:
        mask - default=None

    Returns
    -------------
    peroxi_object
        mask defined extent of peroxisome object
    """
    filament_scale = 1
    filament_cut = 0.15
    small_obj_w = 2
    return infer_endoplasmic_reticulum(in_img, cytosol_mask, filament_scale, filament_cut, small_obj_w)

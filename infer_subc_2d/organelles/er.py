import numpy as np

from infer_subc_2d.constants import ER_CH

from infer_subc_2d.utils.img import (
    size_filter_linear_size,
    select_channel_from_raw,
    filament_filter,
    normalized_edge_preserving_smoothing
)

##########################
#  infer_endoplasmic_reticulum
##########################
def infer_endoplasmic_reticulum(
    in_img: np.ndarray,
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
    er = normalized_edge_preserving_smoothing(er)

   ###################
    # CORE_PROCESSING
    ###################
    # f2_param = [[filament_scale, filament_cut]]
    # # f2_param = [[1, 0.15]]  # [scale_1, cutoff_1]
    # struct_obj = filament_2d_wrapper(er, f2_param)
    struct_obj = filament_filter(er, filament_scale, filament_cut)

    ###################
    # POST_PROCESSING
    ################### 
    struct_obj = size_filter_linear_size(struct_obj, 
                                                    min_size= small_obj_w)

    return struct_obj


##########################
#  fixed_infer_endoplasmic_reticulum
##########################
def fixed_infer_endoplasmic_reticulum(in_img: np.ndarray) -> np.ndarray:
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
    return infer_endoplasmic_reticulum(in_img, filament_scale, filament_cut, small_obj_w)

import numpy as np

from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import edge_preserving_smoothing_3d

from infer_subc_2d.constants import ER_CH

from infer_subc_2d.utils.img import (
    apply_mask,
    min_max_intensity_normalization,
    size_filter_2D,
)


##########################
#  infer_endoplasmic_reticulum
##########################
def infer_endoplasmic_reticulum(in_img: np.ndarray, cytosol_mask: np.ndarray) -> np.ndarray:
    """
    Procedure to infer peroxisome from linearly unmixed input.

    Parameters:
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels

    cytosol_mask: np.ndarray
        mask of cytosol

    Returns:
    -------------
    peroxi_object
        mask defined extent of peroxisome object
    """

    ###################
    # PRE_PROCESSING
    ###################
    struct_img = min_max_intensity_normalization(in_img[ER_CH].copy())

    # edge-preserving smoothing (Option 2, used for Sec61B)
    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)

    ###################
    # CORE_PROCESSING
    ###################
    ################################
    ## PARAMETERS for this step ##
    f2_param = [[1, 0.15]]
    ################################

    bw = filament_2d_wrapper(struct_img, f2_param)

    ###################
    # POST_PROCESSING
    ###################

    ################################
    ## PARAMETERS for this step ##
    small_object_max = 2
    ################################
    # struct_obj = remove_small_objects(struct_obj>0, min_size=min_area, connectivity=1, in_place=False)
    # out_p["min_area"] = min_area
    struct_obj = apply_mask(bw, cytosol_mask)

    struct_obj = size_filter_2D(struct_obj, min_size=small_object_max**2, connectivity=1)

    return struct_obj

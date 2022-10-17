from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    edge_preserving_smoothing_3d,
)


from infer_subc_2d.utils.img import *

##########################
#  infer_ENDOPLASMIC_RETICULUM
##########################
def infer_ENDOPLASMIC_RETICULUM(struct_img, CY_object, in_params) -> tuple:
    """
    Procedure to infer ER  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 2d image containing the ER signal

    CY_object: np.ndarray boolean
        a 2d (3D with 1 Z) image containing the cytosol mask
    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of ER
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p = in_params.copy()
    struct_img = apply_mask(struct_img, CY_object)

    ###################
    # PRE_PROCESSING
    ###################
    intensity_norm_param = [0]  # CHECK THIS

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    out_p["intensity_norm_param"] = intensity_norm_param

    # edge-preserving smoothing (Option 2, used for Sec61B)
    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)

    ###################
    # CORE_PROCESSING
    ###################
    ################################
    ## PARAMETERS for this step ##
    f2_param = [[1, 0.15]]
    ################################

    struct_obj = filament_2d_wrapper(struct_img, f2_param)
    out_p["f2_param"] = f2_param

    ###################
    # POST_PROCESSING
    ###################

    ################################
    ## PARAMETERS for this step ##
    small_object_max = 2
    ################################
    # struct_obj = remove_small_objects(struct_obj>0, min_size=min_area, connectivity=1, in_place=False)
    # out_p["min_area"] = min_area

    struct_obj = size_filter_2D(struct_obj, min_size=small_object_max**2, connectivity=1)
    out_p["small_object_max"] = small_object_max

    retval = (struct_obj, out_p)
    return retval

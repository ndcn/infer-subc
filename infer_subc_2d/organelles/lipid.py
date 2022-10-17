from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import remove_small_holes  # function for post-processing (size filter)

from aicssegmentation.core.pre_processing_utils import intensity_normalization

from infer_subc_2d.utils.img import *


##########################
#  infer_LIPID_DROPLET
##########################
def infer_LIPID_DROPLET(struct_img, CY_object, in_params) -> tuple:
    """
    Procedure to infer LIPID_DROPLET  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 2d (actually single Z) image containing the LIPID_DROPLET signal

    CY_object: np.ndarray boolean
        a 2d (1Z- 3D) image containing the CYTO

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of LD
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p = in_params.copy()

    struct_img = apply_mask(struct_img, CY_object)

    ###################
    # PRE_PROCESSING
    ###################
    # TODO: replace params below with the input params
    scaling_param = [0]
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param

    med_filter_size = 2
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter(struct_img, size=med_filter_size)
    out_p["median_filter_size"] = med_filter_size

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    struct_img = gaussian_filter(
        struct_img, sigma=gaussian_smoothing_sigma, mode="nearest", truncate=gaussian_smoothing_truncate_range
    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    ###################
    # CORE_PROCESSING
    ###################
    threshold_val = threshold_li(struct_img)

    threshold_factor = 0.99  # from cellProfiler
    thresh_min = 0.5
    thresh_max = 1.0
    threshold = min(max(threshold_val * threshold_factor, thresh_min), thresh_max)
    out_p["threshold_factor"] = threshold_factor
    out_p["thresh_min"] = thresh_min
    out_p["thresh_max"] = thresh_max

    struct_obj = struct_img > threshold

    ###################
    # POST_PROCESSING
    ###################
    hole_width = 2.5
    # # wrapper to remoce_small_objects
    struct_obj = remove_small_holes(struct_obj, hole_width**2)
    out_p["hole_width"] = hole_width

    small_object_max = 4
    struct_obj = size_filter_2D(struct_obj, min_size=small_object_max**2, connectivity=1)
    out_p["small_object_max"] = small_object_max

    retval = (struct_obj, out_p)
    return retval

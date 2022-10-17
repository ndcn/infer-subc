from scipy.ndimage import median_filter

# from napari_aicsimageio.core import  reader_function
from aicssegmentation.core.seg_dot import dot_3d_wrapper
from aicssegmentation.core.utils import topology_preserving_thinning
from aicssegmentation.core.MO_threshold import MO

from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    image_smoothing_gaussian_slice_by_slice,
)


from infer_subc_2d.utils.img import *

##########################
#  infer_GOLGI
##########################
def infer_GOLGI(struct_img, CY_object, in_params) -> tuple:
    """
    Procedure to infer GOLGI COMPLEX  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 2d image containing the GOLGI signal

    CY_object: np.ndarray boolean
        a 2d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of MITOCHONDRIA
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p = in_params.copy()

    struct_img = apply_mask(struct_img, CY_object)

    ###################
    # PRE_PROCESSING
    ###################
    intensity_norm_param = [0.1, 30.0]  # CHECK THIS

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    out_p["intensity_norm_param"] = intensity_norm_param

    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    med_filter_size = 3
    struct_img = median_filter(struct_img, size=med_filter_size)
    out_p["median_filter_size"] = med_filter_size

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(
        struct_img, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    ###################
    # CORE_PROCESSING
    ###################
    cell_wise_min_area = 1200
    bw, object_for_debug = MO(
        struct_img, global_thresh_method="tri", object_minArea=cell_wise_min_area, return_object=True
    )
    out_p["cell_wise_min_area"] = cell_wise_min_area

    thin_dist_preserve = 1.6
    thin_dist = 1
    bw_thin = topology_preserving_thinning(bw > 0, thin_dist_preserve, thin_dist)
    out_p["thin_dist_preserve"] = thin_dist_preserve
    out_p["thin_dist"] = thin_dist

    dot_2d_sigma = 1.6
    dot_2d_cutoff = 0.02
    s3_param = [(dot_2d_sigma, dot_2d_cutoff)]

    bw_extra = dot_3d_wrapper(struct_img, s3_param)
    out_p["dot_2d_sigma"] = dot_2d_sigma
    out_p["dot_2d_cutoff"] = dot_2d_cutoff
    out_p["s3_param"] = s3_param

    bw = np.logical_or(bw_extra > 0, bw_thin)

    ###################
    # POST_PROCESSING
    ###################

    small_object_max = 4
    struct_obj = size_filter_2D(bw, min_size=small_object_max**2, connectivity=1)
    out_p["small_object_max"] = small_object_max

    retval = (struct_obj, out_p)
    return retval

from skimage.morphology import remove_small_objects, ball, dilation
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from aicssegmentation.core.seg_dot import dot_3d_wrapper
from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    image_smoothing_gaussian_slice_by_slice,
)

from scipy.ndimage import distance_transform_edt


from infer_subc_2d.utils.img import *

##########################
#  infer_peroxisomes
##########################
def infer_PEROXISOME(struct_img, CY_object, in_params, do_watershed=False) -> tuple:
    """
    Procedure to infer PEROXISOME  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 2d image containing the PEROXISOME signal

    CY_object: np.ndarray boolean
        a 2d (3D with 1 Z) image containing the cytosol mask

    in_params: dict
        holds the needed parameters

    do_watersed: bool
        flag to perform watershed to setment / label adjacent/overlapping peroxisomes
    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of PEROXISOME
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

    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(
        struct_img, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    # log_img, d = log_transform( struct_img )
    # struct_img = intensity_normalization( log_img ,  scaling_param=[0] )
    # struct_img = intensity_normalization( struct_img ,  scaling_param=[0] )

    ###################
    # CORE_PROCESSING
    ###################
    dot_2d_sigma = 1.0
    dot_2d_cutoff = 0.04
    s3_param = [(dot_2d_sigma, dot_2d_cutoff)]

    bw = dot_3d_wrapper(struct_img, s3_param)
    out_p["dot_2d_sigma"] = dot_2d_sigma
    out_p["dot_2d_cutoff"] = dot_2d_cutoff
    out_p["s3_param"] = s3_param

    ###################
    # POST_PROCESSING
    ###################
    if do_watershed:  # BUG: this makes bw into labels... but we are returning a binary object...
        minArea = 4
        mask_ = remove_small_objects(bw > 0, min_size=minArea, connectivity=1, in_place=False)
        seed_ = dilation(peak_local_max(struct_img, labels=label(mask_), min_distance=2, indices=False), selem=ball(1))
        watershed_map = -1 * distance_transform_edt(bw)
        bw = watershed(watershed_map, label(seed_), mask=mask_, watershed_line=True)

    ################################
    ## PARAMETERS for this step ##
    small_object_max = 2
    ################################

    # struct_obj = remove_small_objects(bw>0, min_size=minArea, connectivity=1, in_place=False)
    struct_obj = size_filter_2D(bw, min_size=small_object_max**2, connectivity=1)
    out_p["small_object_max"] = small_object_max

    retval = (struct_obj, out_p)
    return retval

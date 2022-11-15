import numpy as np

from aicssegmentation.core.seg_dot import dot_3d_wrapper
from aicssegmentation.core.utils import topology_preserving_thinning
from aicssegmentation.core.MO_threshold import MO

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice


from aicssegmentation.core.pre_processing_utils import (
    image_smoothing_gaussian_slice_by_slice,
)

from infer_subc_2d.constants import GOLGI_CH

from infer_subc_2d.utils.img import (
    apply_mask,
    median_filter_slice_by_slice,
    min_max_intensity_normalization,
    size_filter_2D,
)


##########################
#  infer_golgi
##########################
def infer_golgi(in_img: np.ndarray, cytosol_mask: np.ndarray) -> np.ndarray:
    """
     Procedure to infer golgi from linearly unmixed input.

     Parameters:
     ------------
     in_img: np.ndarray
         a 3d image containing all the channels

     cytosol_mask: np.ndarray
         mask of cytosol

     Returns:
     -------------
    golgi_object
         mask defined extent of golgi object
    """

    ###################
    # PRE_PROCESSING
    ###################
    struct_img = min_max_intensity_normalization(in_img[GOLGI_CH].copy())

    med_filter_size = 3
    struct_img = median_filter_slice_by_slice(struct_img, size=med_filter_size)

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(
        struct_img, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )

    ###################
    # CORE_PROCESSING
    ###################
    cell_wise_min_area = 1200
    bw, object_for_debug = MO(
        struct_img, global_thresh_method="tri", object_minArea=cell_wise_min_area, return_object=True
    )

    thin_dist_preserve = 1.6
    thin_dist = 1
    bw_thin = topology_preserving_thinning(bw, thin_dist_preserve, thin_dist)

    dot_2d_sigma = 1.6
    dot_2d_cutoff = 0.02
    s3_param = [(dot_2d_sigma, dot_2d_cutoff)]

    bw_extra = dot_3d_wrapper(struct_img, s3_param)

    bw = np.logical_or(bw_extra, bw_thin)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = apply_mask(bw, cytosol_mask)

    small_object_width = 3
    struct_obj = size_filter_2D(struct_obj, min_size=small_object_width**2, connectivity=1)

    return struct_obj

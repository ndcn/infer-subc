import numpy as np

from aicssegmentation.core.pre_processing_utils import (
    image_smoothing_gaussian_slice_by_slice,
)

from infer_subc_2d.constants import MITO_CH

from infer_subc_2d.utils.img import (
    apply_mask,
    median_filter_slice_by_slice,
    min_max_intensity_normalization,
    size_filter_2D,
    vesselness_slice_by_slice,
)

##########################
#  infer_mitochondria
##########################
def infer_mitochondria(in_img: np.ndarray, cytosol_mask: np.ndarray) -> np.ndarray:
    """
    Procedure to infer mitochondria from linearly unmixed input.

    Parameters:
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels

    cytosol_mask: np.ndarray
        mask of cytosol

    Returns:
    -------------
    lysosome_object
        mask defined extent of mitochondria object

    """

    ###################
    # PRE_PROCESSING
    ###################
    struct_img = min_max_intensity_normalization(in_img[MITO_CH].copy())

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
    ################################
    ## PARAMETERS for this step ##
    vesselness_sigma = [1.5]
    vesselness_cutoff = 0.1
    # 2d vesselness slice by slice
    bw = vesselness_slice_by_slice(struct_img, sigmas=vesselness_sigma, cutoff=vesselness_cutoff, tau=0.75)

    ###################
    # POST_PROCESSING
    ###################

    struct_obj = apply_mask(bw, cytosol_mask)

    small_object_max = 3
    struct_obj = size_filter_2D(struct_obj, min_size=small_object_max**2, connectivity=1)

    return struct_obj

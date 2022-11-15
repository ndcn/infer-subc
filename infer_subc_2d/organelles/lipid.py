import numpy as np

from skimage.morphology import remove_small_holes  # function for post-processing (size filter)

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.utils import hole_filling

from infer_subc_2d.utils.img import (
    apply_threshold,
    apply_mask,
    min_max_intensity_normalization,
    median_filter_slice_by_slice,
    size_filter_2D,
)
from infer_subc_2d.constants import LIPID_CH


##########################
#  infer_lipid_body
##########################
##########################
#  infer_endoplasmic_reticulum
##########################
def infer_lipid_body(in_img: np.ndarray, cytosol_mask: np.ndarray) -> np.ndarray:
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
    struct_img = min_max_intensity_normalization(in_img[LIPID_CH].copy())

    med_filter_size = 2
    struct_img = median_filter_slice_by_slice(struct_img, size=med_filter_size)

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(
        struct_img, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )

    ###################
    # CORE_PROCESSING
    ###################

    threshold_factor = 0.99  # from cellProfiler
    thresh_min = 0.5
    thresh_max = 1.0
    bw = apply_threshold(
        struct_img, method="otsu", threshold_factor=threshold_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )

    ###################
    # POST_PROCESSING
    ###################

    # 2D cleaning
    hole_min = 0
    hole_max = 2.5
    struct_obj = hole_filling(bw, hole_min=hole_min**2, hole_max=hole_max**2, fill_2d=True)

    struct_obj = apply_mask(struct_obj, cytosol_mask)

    small_object_width = 4

    struct_obj = size_filter_2D(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_width**2,
        connectivity=1,
    )

    return struct_obj

import numpy as np

from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice

from infer_subc_2d.constants import LYSO_CH

from infer_subc_2d.utils.img import (
    apply_mask,
    median_filter_slice_by_slice,
    min_max_intensity_normalization,
    size_filter_2D,
)

##########################
#  infer_lysosomes
##########################
def infer_lysosomes(in_img: np.ndarray, cytosol_mask: np.ndarray) -> np.ndarray:
    """
    Procedure to infer lysosome from linearly unmixed input.

    Parameters:
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels

    cytosol_mask: np.ndarray
        mask of cytosol

    Returns:
    -------------
    lysosome_object
        mask defined extent of lysosome object
    """

    ###################
    # PRE_PROCESSING
    ###################
    struct_img = min_max_intensity_normalization(in_img[LYSO_CH].copy())

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
    # dot and filiment enhancement - 2D
    s2_param = [[5, 0.09], [2.5, 0.07], [1, 0.01]]
    bw_spot = dot_2d_slice_by_slice_wrapper(struct_img, s2_param)

    f2_param = [[1, 0.15]]
    bw_filament = filament_2d_wrapper(struct_img, f2_param)

    bw = np.logical_or(bw_spot, bw_filament)
    # bw = segmentation_union([bw_spot,bw_filament])

    ###################
    # POST_PROCESSING
    ###################
    # 2D cleaning
    hole_min = 0
    hole_max = 25
    struct_obj = hole_filling(bw, hole_min=hole_min**2, hole_max=hole_max**2, fill_2d=True)

    struct_obj = apply_mask(struct_obj, cytosol_mask)

    small_object_width = 3
    struct_obj = size_filter_2D(struct_obj, min_size=small_object_width**2, connectivity=1)

    return struct_obj

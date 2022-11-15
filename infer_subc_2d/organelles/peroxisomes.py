import numpy as np

from skimage.morphology import remove_small_objects, ball, dilation
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper

from infer_subc_2d.constants import PEROXI_CH

from infer_subc_2d.utils.img import (
    apply_mask,
    min_max_intensity_normalization,
    size_filter_2D,
)


##########################
#  infer_peroxisome
##########################
def infer_peroxisome(in_img: np.ndarray, cytosol_mask: np.ndarray) -> np.ndarray:
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
    struct_img = min_max_intensity_normalization(in_img[PEROXI_CH].copy())

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(
        struct_img, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )
    ###################
    # CORE_PROCESSING
    ###################
    dot_2d_sigma = 1.0
    dot_2d_cutoff = 0.04
    s2_param = [(dot_2d_sigma, dot_2d_cutoff)]

    bw = dot_2d_slice_by_slice_wrapper(struct_img, s2_param)

    ###################
    # POST_PROCESSING
    ###################
    do_watershed = False
    if do_watershed:  # BUG: this makes bw into labels... but we are returning a binary object...
        minArea = 4
        mask_ = remove_small_objects(bw > 0, min_size=minArea, connectivity=1, in_place=False)
        seed_ = dilation(peak_local_max(struct_img, labels=label(mask_), min_distance=2, indices=False), selem=ball(1))
        watershed_map = -1 * distance_transform_edt(bw)
        bw = watershed(watershed_map, label(seed_), mask=mask_, watershed_line=True)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = apply_mask(bw, cytosol_mask)

    small_object_width = 2
    struct_obj = size_filter_2D(struct_obj, min_size=small_object_width**2, connectivity=1)

    return struct_obj

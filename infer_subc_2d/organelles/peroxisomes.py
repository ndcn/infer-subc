from scipy.ndimage import median_filter, extrema, distance_transform_edt
from scipy.interpolate import RectBivariateSpline

from skimage import img_as_float, filters
from skimage import morphology
from skimage.morphology import remove_small_objects, ball, disk, dilation, binary_closing, white_tophat, black_tophat
from skimage.filters import threshold_triangle, threshold_otsu, threshold_li, threshold_multiotsu, threshold_sauvola
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# from napari_aicsimageio.core import  reader_function
import aicssegmentation
from aicssegmentation.core.seg_dot import dot_3d_wrapper, dot_slice_by_slice, dot_2d_slice_by_slice_wrapper, dot_3d

from aicssegmentation.core.utils import topology_preserving_thinning, hole_filling, size_filter
from aicssegmentation.core.MO_threshold import MO
from aicssegmentation.core.vessel import filament_2d_wrapper, vesselnessSliceBySlice
from aicssegmentation.core.output_utils import save_segmentation, generate_segmentation_contour
from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    image_smoothing_gaussian_3d,
    image_smoothing_gaussian_slice_by_slice,
    edge_preserving_smoothing_3d,
)


from infer_subc_2d.utils.img import *

##########################
#  infer_PEROXISOME
##########################
def infer_PEROXISOME(struct_img, CY_object, in_params) -> tuple:
    """
    Procedure to infer PEROXISOME  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the PEROXISOME signal

    CY_object: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of PEROXISOME
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p = in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################
    intensity_norm_param = [0]  # CHECK THIS

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    out_p["intensity_norm_param"] = intensity_norm_param

    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    gaussian_smoothing_sigma = 1.0
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_3d(
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
    dot_3d_sigma = 1.0
    dot_3d_cutoff = 0.04
    s3_param = [(dot_3d_sigma, dot_3d_cutoff)]

    bw = dot_3d_wrapper(struct_img, s3_param)
    out_p["dot_3d_sigma"] = dot_3d_sigma
    out_p["dot_3d_cutoff"] = dot_3d_cutoff
    out_p["s3_param"] = s3_param

    ###################
    # POST_PROCESSING
    ###################
    # watershed
    minArea = 4
    mask_ = remove_small_objects(bw > 0, min_size=minArea, connectivity=1, in_place=False)
    seed_ = dilation(peak_local_max(struct_img, labels=label(mask_), min_distance=2, indices=False), selem=ball(1))
    watershed_map = -1 * distance_transform_edt(bw)
    struct_obj = watershed(watershed_map, label(seed_), mask=mask_, watershed_line=True)
    ################################
    ## PARAMETERS for this step ##
    min_area = 4
    ################################
    struct_obj = remove_small_objects(struct_obj > 0, min_size=min_area, connectivity=1, in_place=False)
    out_p["min_area"] = min_area

    retval = (struct_obj, out_p)
    return retval

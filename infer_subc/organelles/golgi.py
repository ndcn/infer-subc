from scipy.ndimage import median_filter, extrema
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


from infer_subc.utils.img import *


##########################
#  infer_GOLGI
##########################
def infer_GOLGI(struct_img, CY_object, in_params) -> tuple:
    """
    Procedure to infer GOLGI COMPLEX  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the GOLGI signal

    CY_object: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of GOLGI
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p = in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################
    intensity_norm_param = [0.1, 30.0]  # CHECK THIS

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    out_p["intensity_norm_param"] = intensity_norm_param

    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    med_filter_size = 3
    structure_img = median_filter(struct_img, size=med_filter_size)
    out_p["median_filter_size"] = med_filter_size

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

    dot_3d_sigma = 1.6
    dot_3d_cutoff = 0.02
    s3_param = [(dot_3d_sigma, dot_3d_cutoff)]

    bw_extra = dot_3d_wrapper(struct_img, s3_param)
    out_p["dot_3d_sigma"] = dot_3d_sigma
    out_p["dot_3d_cutoff"] = dot_3d_cutoff
    out_p["s3_param"] = s3_param

    bw = np.logical_or(bw_extra > 0, bw_thin)

    ###################
    # POST_PROCESSING
    ###################

    # MT_object = remove_small_objects(bw > 0, min_size=small_object_max**2, connectivity=1, in_place=False)
    small_object_max = 10
    struct_obj = size_filter(
        bw,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_max**3,
        method="3D",  # "slice_by_slice" ,
        connectivity=1,
    )
    out_p["small_object_max"] = small_object_max

    retval = (struct_obj, out_p)
    return retval

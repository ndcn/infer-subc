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


from infer_subc_2d.utils.img import *


def infer_CELL_MEMBRANE(struct_img, in_params) -> tuple:
    """
    Procedure to infer NUCLEI from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the NUCLEI signal

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of NU
        label
            label (could be more than 1)
        signal
            scaled/filtered (pre-processed) flourescence image
        parameters: dict
            updated parameters in case any needed were missing

    """

    out_p = in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################

    # TODO: replace params below with the input params
    scaling_param = [0]
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param

    med_filter_size = 4
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter_slice_by_slice(struct_img, size=med_filter_size)
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

    struct_obj = struct_img > filters.threshold_li(struct_img)
    threshold_value_log = threshold_li_log(struct_img)

    thresh_factor = 0.9  # from cellProfiler
    thresh_min = 0.1
    thresh_max = 1.0
    threshold = min(max(threshold_value_log * thresh_factor, thresh_min), thresh_max)
    out_p["thresh_factor"] = thresh_factor
    out_p["thresh_min"] = thresh_min
    out_p["thresh_max"] = thresh_max

    struct_obj = struct_img > threshold

    ###################
    # POST_PROCESSING
    ###################

    hole_width = 5
    # # wrapper to remoce_small_objects
    struct_obj = morphology.remove_small_holes(struct_obj, hole_width**3)
    out_p["hole_width"] = hole_width

    small_object_width = 5
    struct_obj = aicssegmentation.core.utils.size_filter(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_width**3,
        method="slice_by_slice",  # "3D", #
        connectivity=1,
    )
    out_p["small_object_width"] = small_object_width

    retval = (struct_obj, label(struct_obj), out_p)
    return retval

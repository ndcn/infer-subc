from scipy.ndimage import median_filter, extrema, sobel
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
# 2a.  infer_SOMA1
##########################
def infer_SOMA1(struct_img: np.ndarray, NU_labels: np.ndarray, in_params: dict) -> tuple:
    """
    Procedure to infer SOMA from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the SOMA signal

    NU_labels: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of SOMA
        label
            label (could be more than 1)
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

    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    # Linear-ish processing
    med_filter_size = 15
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter_slice_by_slice(struct_img, size=med_filter_size)
    out_p["median_filter_size"] = med_filter_size
    gaussian_smoothing_sigma = 1.0
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(
        struct_img, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    # non-Linear processing
    log_img, d = log_transform(struct_img)
    log_img = intensity_normalization(log_img, scaling_param=[0])

    struct_img = intensity_normalization(filters.scharr(log_img), scaling_param=[0]) + log_img

    ###################
    # CORE_PROCESSING
    ###################
    local_adjust = 0.5
    low_level_min_size = 100
    # "Masked Object Thresholding" - 3D
    struct_obj, _bw_low_level = MO(
        struct_img,
        global_thresh_method="ave",
        object_minArea=low_level_min_size,
        extra_criteria=True,
        local_adjust=local_adjust,
        return_object=True,
        dilate=True,
    )

    out_p["local_adjust"] = local_adjust
    out_p["low_level_min_size"] = low_level_min_size

    ###################
    # POST_PROCESSING
    ###################

    # 2D
    hole_max = 80
    struct_obj = hole_filling(struct_obj, hole_min=0.0, hole_max=hole_max**2, fill_2d=True)
    out_p["hole_max"] = hole_max

    small_object_max = 35
    struct_obj = size_filter(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_max**3,
        method="slice_by_slice",
        connectivity=1,
    )
    out_p["small_object_max"] = small_object_max

    labels_out = watershed(
        connectivity=np.ones((3, 3, 3), bool),
        image=1.0 - struct_img,
        markers=NU_labels,
        mask=np.logical_or(struct_obj, NU_labels > 0),
    )
    ###################
    # POST- POST_PROCESSING
    ###################
    # keep the "SOMA" label which contains the highest total signal
    all_labels = np.unique(labels_out)[1:]

    total_signal = [scaled_signal[labels_out == label].sum() for label in all_labels]
    # combine NU and "labels" to make a SOMA
    keep_label = all_labels[np.argmax(total_signal)]

    # now use all the NU labels which AREN't keep_label and add to mask and re-label
    masked_composite_soma = struct_img.copy()
    new_NU_mask = np.logical_and(NU_labels != 0, NU_labels != keep_label)

    # "Masked Object Thresholding" - 3D
    masked_composite_soma[new_NU_mask] = 0
    struct_obj, _bw_low_level = MO(
        masked_composite_soma,
        global_thresh_method="ave",
        object_minArea=low_level_min_size,
        extra_criteria=True,
        local_adjust=local_adjust,
        return_object=True,
        dilate=True,
    )

    struct_obj = hole_filling(struct_obj, hole_min=0.0, hole_max=hole_max**2, fill_2d=True)
    struct_obj = size_filter(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_max**3,
        method="slice_by_slice",
        connectivity=1,
    )
    masked_labels_out = watershed(
        connectivity=np.ones((3, 3, 3), bool),
        image=1.0 - struct_img,
        markers=NU_labels,
        mask=np.logical_or(struct_obj, NU_labels == keep_label),
    )

    retval = (struct_obj, masked_labels_out, out_p)
    return retval


##########################
# 2b.  infer_SOMA2
########################### copy this to base.py for easy import
def infer_SOMA2(struct_img: np.ndarray, NU_labels: np.ndarray, in_params: dict) -> tuple:
    """
    Procedure to infer SOMA from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the SOMA signal

    NU_labels: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of SOMA
        label
            label (could be more than 1)
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

    # 2D smoothing
    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    med_filter_size = 9
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter_slice_by_slice(struct_img, size=med_filter_size)
    out_p["median_filter_size"] = med_filter_size

    gaussian_smoothing_sigma = 3.0
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(
        struct_img, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    #    edges = filters.scharr(struct_img)
    # struct_img, d = log_transform( struct_img )
    # struct_img = intensity_normalization(  struct_img,  scaling_param=[0] )
    ###################
    # CORE_PROCESSING
    ###################
    # "Masked Object Thresholding" - 3D
    local_adjust = 0.25
    low_level_min_size = 100
    struct_obj, _bw_low_level = MO(
        struct_img,
        global_thresh_method="ave",
        object_minArea=low_level_min_size,
        extra_criteria=True,
        local_adjust=local_adjust,
        return_object=True,
        dilate=True,
    )
    out_p["local_adjust"] = local_adjust
    out_p["low_level_min_size"] = low_level_min_size

    ###################
    # POST_PROCESSING
    ###################
    # 3D cleaning

    hole_max = 80
    # discount z direction
    struct_obj = hole_filling(struct_obj, hole_min=0.0, hole_max=hole_max**2, fill_2d=True)
    out_p["hole_max"] = hole_max

    small_object_max = 35
    struct_obj = size_filter(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_max**3,
        method="3D",  # "slice_by_slice"
        connectivity=1,
    )
    out_p["small_object_max"] = small_object_max

    labels_out = watershed(
        image=np.abs(sobel(struct_img)),
        markers=NU_labels,
        connectivity=np.ones((3, 3, 3), bool),
        mask=np.logical_or(struct_obj, NU_labels > 0),
    )

    ###################
    # POST- POST_PROCESSING
    ###################
    # keep the "SOMA" label which contains the highest total signal
    all_labels = np.unique(labels_out)[1:]

    total_signal = [scaled_signal[labels_out == label].sum() for label in all_labels]
    # combine NU and "labels" to make a SOMA
    keep_label = all_labels[np.argmax(total_signal)]

    # now use all the NU labels which AREN't keep_label and add to mask and re-label
    masked_composite_soma = struct_img.copy()
    new_NU_mask = np.logical_and(NU_labels != 0, NU_labels != keep_label)

    # "Masked Object Thresholding" - 3D
    masked_composite_soma[new_NU_mask] = 0
    struct_obj, _bw_low_level = MO(
        masked_composite_soma,
        global_thresh_method="ave",
        object_minArea=low_level_min_size,
        extra_criteria=True,
        local_adjust=local_adjust,
        return_object=True,
        dilate=True,
    )
    # 3D cleaning
    struct_obj = hole_filling(struct_obj, hole_min=0.0, hole_max=hole_max**2, fill_2d=True)
    struct_obj = size_filter(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_max**3,
        method="3D",  # "slice_by_slice"
        connectivity=1,
    )
    masked_labels_out = watershed(
        connectivity=np.ones((3, 3, 3), bool),
        image=np.abs(sobel(struct_img)),
        markers=NU_labels,
        mask=np.logical_or(struct_obj, NU_labels == keep_label),
    )

    retval = (struct_obj, masked_labels_out, out_p)
    return retval


##########################
# 2c.  infer_SOMA3
##########################
def infer_SOMA3(struct_img, NU_labels, in_params) -> tuple:
    """
    Procedure to infer SOMA from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the SOMA signal

    NU_labels: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of SOMA
        label
            label (could be more than 1)
        parameters: dict
            updated parameters in case any needed were missing

    """
    out_p = in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################
    scaling_param = [0]
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param

    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    med_filter_size = 3
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter_slice_by_slice(struct_img, size=med_filter_size)
    out_p["median_filter_size"] = med_filter_size

    gaussian_smoothing_sigma = 1.0
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(
        struct_img, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    log_img, d = log_transform(struct_img)
    struct_img = intensity_normalization(log_img + filters.scharr(log_img), scaling_param=[0])

    ###################
    # CORE_PROCESSING
    ###################
    # "Masked Object Thresholding" - 3D
    local_adjust = 0.5
    low_level_min_size = 100
    struct_obj, _bw_low_level = MO(
        struct_img,
        global_thresh_method="ave",
        object_minArea=low_level_min_size,
        extra_criteria=True,
        local_adjust=local_adjust,
        return_object=True,
        dilate=True,
    )
    out_p["local_adjust"] = local_adjust
    out_p["low_level_min_size"] = low_level_min_size
    ###################
    # POST_PROCESSING
    ###################
    # 2D cleaning
    hole_max = 100
    # discount z direction
    struct_obj = hole_filling(struct_obj, hole_min=0.0, hole_max=hole_max**2, fill_2d=True)
    out_p["hole_max"] = hole_max

    small_object_max = 30
    struct_obj = size_filter(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_max**3,
        method="slice_by_slice",
        connectivity=1,
    )
    out_p["small_object_max"] = small_object_max

    labels_out = watershed(
        image=np.abs(
            sobel(struct_img)
        ),  # either log_img or struct_img seem to work, but more spurious labeling to fix in post-post for struct_img
        markers=NU_labels,
        connectivity=np.ones((3, 3, 3), bool),
        mask=np.logical_or(struct_obj, NU_labels > 0),
    )

    ###################
    # POST- POST_PROCESSING
    ###################
    # keep the "SOMA" label which contains the highest total signal
    all_labels = np.unique(labels_out)[1:]

    total_signal = [scaled_signal[labels_out == label].sum() for label in all_labels]
    # combine NU and "labels" to make a SOMA
    keep_label = all_labels[np.argmax(total_signal)]

    # now use all the NU labels which AREN't keep_label and add to mask and re-label
    masked_composite_soma = struct_img.copy()
    new_NU_mask = np.logical_and(NU_labels != 0, NU_labels != keep_label)

    # "Masked Object Thresholding" - 3D
    masked_composite_soma[new_NU_mask] = 0
    struct_obj, _bw_low_level = MO(
        masked_composite_soma,
        global_thresh_method="ave",
        object_minArea=low_level_min_size,
        extra_criteria=True,
        local_adjust=local_adjust,
        return_object=True,
        dilate=True,
    )

    # 2D cleaning
    struct_obj = hole_filling(struct_obj, hole_min=0.0, hole_max=hole_max**2, fill_2d=True)
    struct_obj = size_filter(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_max**3,
        method="slice_by_slice",
        connectivity=1,
    )
    masked_labels_out = watershed(
        connectivity=np.ones((3, 3, 3), bool),
        image=np.abs(
            sobel(struct_img)
        ),  # either log_img or struct_img seem to work, but more spurious labeling to fix in post-post for struct_img
        markers=NU_labels,
        mask=np.logical_or(struct_obj, NU_labels == keep_label),
    )

    retval = (struct_obj, masked_labels_out, out_p)
    return retval

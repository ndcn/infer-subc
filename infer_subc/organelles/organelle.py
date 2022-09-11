from imports import *
from infer_subc.utils.img import *


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


##########################
# 1.  infer_NUCLEI
##########################
# copy this to base.py for easy import


def infer_NUCLEI(struct_img, in_params) -> tuple:
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

    threshold_factor = 0.9  # from cellProfiler
    thresh_min = 0.1
    thresh_max = 1.0
    threshold = min(max(threshold_value_log * threshold_factor, thresh_min), thresh_max)
    out_p["threshold_factor"] = threshold_factor
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

    small_object_max = 5
    struct_obj = aicssegmentation.core.utils.size_filter(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_max**3,
        method="slice_by_slice",  # "3D", #
        connectivity=1,
    )
    out_p["small_object_max"] = small_object_max

    retval = (struct_obj, label(struct_obj), out_p)
    return retval


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

    threshold_factor = 0.9  # from cellProfiler
    thresh_min = 0.1
    thresh_max = 1.0
    threshold = min(max(threshold_value_log * threshold_factor, thresh_min), thresh_max)
    out_p["threshold_factor"] = threshold_factor
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

    small_object_max = 5
    struct_obj = aicssegmentation.core.utils.size_filter(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_max**3,
        method="slice_by_slice",  # "3D", #
        connectivity=1,
    )
    out_p["small_object_max"] = small_object_max

    retval = (struct_obj, label(struct_obj), out_p)
    return retval


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
        image=np.abs(ndi.sobel(struct_img)),
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
        image=np.abs(ndi.sobel(struct_img)),
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
            ndi.sobel(struct_img)
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
            ndi.sobel(struct_img)
        ),  # either log_img or struct_img seem to work, but more spurious labeling to fix in post-post for struct_img
        markers=NU_labels,
        mask=np.logical_or(struct_obj, NU_labels == keep_label),
    )

    retval = (struct_obj, masked_labels_out, out_p)
    return retval


##########################
#  infer_CYTOSOL
##########################
def infer_CYTOSOL(SO_object, NU_object, erode_NU=True):
    """
    Procedure to infer CYTOSOL from linearly unmixed input.

    Parameters:
    ------------
    SO_object: np.ndarray
        a 3d image containing the NUCLEI signal

    NU_object: np.ndarray
        a 3d image containing the NUCLEI signal

    erode_NU: bool
        should we erode?

    Returns:
    -------------
    CY_object: np.ndarray (bool)

    """

    # NU_eroded1 = morphology.binary_erosion(NU_object,  footprint=morphology.ball(3) )
    if erode_NU:
        CY_object = np.logical_and(SO_object, ~morphology.binary_erosion(NU_object))
    else:
        CY_object = np.logical_and(SO_object, ~NU_object)
    return CY_object


##########################
#  infer_LYSOSOMES
##########################
def infer_LYSOSOMES(struct_img, CY_object, in_params) -> tuple:
    """
    Procedure to infer LYSOSOME from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the LYSOSOME signal

    CY_object: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of Lysosomes
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

    # log_img, d = log_transform( struct_img )
    # struct_img = intensity_normalization( log_img ,  scaling_param=[0] )
    struct_img = intensity_normalization(struct_img, scaling_param=[0])

    ###################
    # CORE_PROCESSING
    ###################
    # dot and filiment enhancement - 2D

    ################################
    ## PARAMETERS for this step ##
    s2_param = [[5, 0.09], [2.5, 0.07], [1, 0.01]]
    ################################
    bw_spot = dot_2d_slice_by_slice_wrapper(struct_img, s2_param)

    ################################
    ## PARAMETERS for this step ##
    f2_param = [[1, 0.15]]
    ################################
    bw_filament = filament_2d_wrapper(struct_img, f2_param)

    bw = np.logical_or(bw_spot, bw_filament)

    out_p["s2_param"] = s2_param
    out_p["f2_param"] = f2_param
    ###################
    # POST_PROCESSING
    ###################

    # 2D cleaning
    hole_max = 1600
    # discount z direction
    struct_obj = aicssegmentation.core.utils.hole_filling(bw, hole_min=0.0, hole_max=hole_max**2, fill_2d=True)
    out_p["hole_max"] = hole_max

    # # 3D
    # cleaned_img = remove_small_objects(removed_holes>0,
    #                                                             min_size=width,
    #                                                             connectivity=1,
    #                                                             in_place=False)

    small_object_max = 15
    struct_obj = aicssegmentation.core.utils.size_filter(
        struct_obj,  # wrapper to remove_small_objects which can do slice by slice
        min_size=small_object_max**3,
        method="3D",  # "slice_by_slice" ,
        connectivity=1,
    )
    out_p["small_object_max"] = small_object_max

    retval = (struct_obj, out_p)
    return retval


##########################
#  infer_MITOCHONDRIA
##########################
def infer_MITOCHONDRIA(struct_img, CY_object, in_params) -> tuple:
    """
    Procedure to infer MITOCHONDRIA  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the MITOCHONDRIA signal

    CY_object: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of MITOCHONDRIA
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p = in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################
    scaling_param = [0, 9]
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param

    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    med_filter_size = 3
    structure_img = median_filter(struct_img, size=med_filter_size)
    out_p["median_filter_size"] = med_filter_size

    gaussian_smoothing_sigma = 1.3
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
    ################################
    ## PARAMETERS for this step ##
    vesselness_sigma = [1.5]
    vesselness_cutoff = 0.16
    # 2d vesselness slice by slice
    response = vesselnessSliceBySlice(struct_img, sigmas=vesselness_sigma, tau=1, whiteonblack=True)
    bw = response > vesselness_cutoff

    out_p["vesselness_sigma"] = vesselness_sigma
    out_p["vesselness_cutoff"] = vesselness_cutoff

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
    watershed_map = -1 * ndi.distance_transform_edt(bw)
    struct_obj = watershed(watershed_map, label(seed_), mask=mask_, watershed_line=True)
    ################################
    ## PARAMETERS for this step ##
    min_area = 4
    ################################
    struct_obj = remove_small_objects(struct_obj > 0, min_size=min_area, connectivity=1, in_place=False)
    out_p["min_area"] = min_area

    retval = (struct_obj, out_p)
    return retval


##########################
#  infer_ENDOPLASMIC_RETICULUM
##########################
def infer_ENDOPLASMIC_RETICULUM(struct_img, CY_object, in_params) -> tuple:
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

    # edge-preserving smoothing (Option 2, used for Sec61B)
    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)

    ###################
    # CORE_PROCESSING
    ###################
    ################################
    ## PARAMETERS for this step ##
    f2_param = [[1, 0.15]]
    ################################

    struct_obj = filament_2d_wrapper(struct_img, f2_param)
    out_p["f2_param"] = f2_param

    ###################
    # POST_PROCESSING
    ###################

    ################################
    ## PARAMETERS for this step ##
    min_area = 20
    ################################
    struct_obj = remove_small_objects(struct_obj > 0, min_size=min_area, connectivity=1, in_place=False)
    out_p["min_area"] = min_area

    retval = (struct_obj, out_p)
    return retval


def infer_LIPID_DROPLET(struct_img, out_path, cyto_labels, in_params):
    pass

from skimage.filters import scharr
from skimage.segmentation import watershed
from skimage.measure import label

# from napari_aicsimageio.core import  reader_function

from aicssegmentation.core.utils import hole_filling, size_filter
from aicssegmentation.core.MO_threshold import MO

from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    image_smoothing_gaussian_slice_by_slice,
)

from infer_subc_2d.constants import (
    TEST_IMG_N,
    NUC_CH,
    LYSO_CH,
    MITO_CH,
    GOLGI_CH,
    PEROXI_CH,
    ER_CH,
    LIPID_CH,
    RESIDUAL_CH,
)
from infer_subc_2d.utils.img import *


def raw_soma_MCZ(img_in):
    """define soma image"""
    SOMA_W = (4.0, 1.0, 1.0)
    SOMA_CH = (LYSO_CH, ER_CH, RESIDUAL_CH)  # LYSO, ER, RESIDUAL
    img_out = np.zeros_like(img_in[0]).astype(np.double)
    for w, ch in zip(SOMA_W, SOMA_CH):
        img_out += w * img_in[ch]
    return img_out


def masked_inverted_watershed(img_in, markers, mask):
    """wrapper for watershed on inverted image and masked"""

    labels_out = watershed(
        1.0 - img_in,
        markers=markers,
        connectivity=np.ones((1, 3, 3), bool),
        mask=mask,
    )
    return labels_out


def non_linear_soma_transform_MCZ(in_img):
    """non-linear distortion to fill out soma
    log + edge of smoothed composite
    """

    # non-Linear processing
    log_img, d = log_transform(in_img.copy())
    log_img = min_max_intensity_normalization(log_img)
    return min_max_intensity_normalization(scharr(log_img)) + log_img


##########################
# 1  infer_SOMA
##########################
def infer_SOMA(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer SOMA from linearly unmixed input.

    Parameters:
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels


    Returns:
    -------------
    soma_out:
          label (could be more than 1)

    """

    struct_img = raw_soma_MCZ(in_img)

    ###################
    # PRE_PROCESSING
    ###################
    ################# part 1- soma

    struct_img = min_max_intensity_normalization(struct_img)

    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    # Linear-ish processing
    med_filter_size = 15
    struct_img = median_filter_slice_by_slice(struct_img, size=med_filter_size)

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(
        struct_img,
        sigma=gaussian_smoothing_sigma,
        truncate_range=gaussian_smoothing_truncate_range,
    )

    struct_img_non_lin = non_linear_soma_transform_MCZ(struct_img)

    ################# part 2 - nuclei
    nuclei = min_max_intensity_normalization(in_img[NUC_CH].copy())

    med_filter_size = 4
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    nuclei = median_filter_slice_by_slice(nuclei, size=med_filter_size)

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    nuclei = image_smoothing_gaussian_slice_by_slice(
        nuclei, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )

    ###################
    # CORE_PROCESSING
    ###################
    local_adjust = 0.5
    low_level_min_size = 100
    # "Masked Object Thresholding" - 3D capable
    struct_obj, _bw_low_level = MO(
        struct_img_non_lin,
        global_thresh_method="ave",
        object_minArea=low_level_min_size,
        extra_criteria=True,
        local_adjust=local_adjust,
        return_object=True,
        dilate=True,
    )

    ################# part 2 : nuclei thresholding
    # struct_obj = struct_img > filters.threshold_li(struct_img)
    threshold_factor = 0.9  # from cellProfiler
    thresh_min = 0.1
    thresh_max = 1.0
    NU_object = apply_log_li_threshold(
        nuclei, threshold_factor=threshold_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )
    hole_width = 5
    # # wrapper to remoce_small_objects
    # NU_object = remove_small_holes(NU_object, hole_width**2)
    NU_object = hole_filling(NU_object, hole_min=0, hole_max=hole_width**2, fill_2d=True)

    small_object_max = 45
    NU_object = size_filter_2D(NU_object, min_size=small_object_max**2, connectivity=1)

    NU_labels = label(NU_object)
    ###################
    # POST_PROCESSING
    ###################

    # 2D
    hole_max = 40
    struct_obj = hole_filling(struct_obj, hole_min=0, hole_max=hole_max**2, fill_2d=True)
    # struct_obj = remove_small_holes(struct_obj, hole_max ** 2 )

    small_object_max = 45
    struct_obj = size_filter_2D(struct_obj, min_size=small_object_max**2, connectivity=1)

    labels_out = masked_inverted_watershed(
        struct_img, NU_labels, struct_obj
    )  # np.logical_or(struct_obj, NU_labels > 0)

    ###################
    # POST- POST_PROCESSING
    ###################
    # keep the "SOMA" label which contains the highest total signal
    soma_out = choose_max_label(struct_img, labels_out)

    return soma_out

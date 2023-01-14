from skimage.filters import scharr
from skimage.segmentation import watershed
from skimage.measure import label

import numpy as np

# from napari_aicsimageio.core import  reader_function

from aicssegmentation.core.utils import hole_filling

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice

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

from infer_subc_2d.utils.img import (
    masked_object_thresh,
    log_transform,
    min_max_intensity_normalization,
    median_filter_slice_by_slice,
    apply_log_li_threshold,
    size_filter_2D,
    choose_max_label,
    select_channel_from_raw,
)


def raw_soma_MCZ(img_in):
    """define soma image
    uses pre-defined weights and channels

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    """
    SOMA_W = (4.0, 1.0, 1.0)
    SOMA_CH = (LYSO_CH, ER_CH, RESIDUAL_CH)  # LYSO, ER, RESIDUAL
    img_out = np.zeros_like(img_in[0]).astype(np.double)
    for w, ch in zip(SOMA_W, SOMA_CH):
        img_out += w * img_in[ch]
    return img_out


def masked_inverted_watershed(img_in, markers, mask):
    """wrapper for watershed on inverted image and masked

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    """

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

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    """

    # non-Linear processing
    log_img, d = log_transform(in_img.copy())
    log_img = min_max_intensity_normalization(log_img)
    return min_max_intensity_normalization(scharr(log_img)) + log_img


##########################
# 1. infer_soma
##########################
def infer_soma(
    in_img: np.ndarray,
    median_sz_soma: int,
    gauss_sig_soma: float,
    median_sz_nuc: int,
    gauss_sig_nuc: float,
    mo_method: str,
    mo_adjust: float,
    mo_cutoff_size: int,
    thresh_factor: float,
    thresh_min: float,
    thresh_max: float,
    max_hole_w_nuc: int,
    small_obj_w_nuc: int,
    max_hole_w_soma: int,
    small_obj_w_soma: int,
) -> np.ndarray:
    """
    Procedure to infer soma from linearly unmixed input.

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    median_sz_soma:
        width of median filter for _soma_ signal
    gauss_sig_soma:
        sigma for gaussian smoothing of _soma_ signal
    median_sz_nuc:
        width of median filter for _soma_ signal
    gauss_sig_nuc:
        sigma for gaussian smoothing of _soma_ signal
    mo_method:
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
    mo_adjust:
        Masked Object threshold `local_adjust`
    mo_cutoff_size:
        Masked Object threshold `size_min`
    thresh_factor:
        adjustment factor for log Li threholding
    thresh_min:
        abs min threhold for log Li threholding
    thresh_max:
        abs max threhold for log Li threholding
    max_hole_w_nuc:
        hole filling cutoff for nuclei post-processing
    small_obj_w_nuc:
        minimu object size cutoff for nuclei post-processing
    max_hole_w_soma:
        hole filling cutoff for soma signal post-processing
    small_obj_w_soma:
        minimu object size cutoff for soma signal post-processing

    Returns
    -------------
    soma_mask:
        a logical/labels object defining boundaries of soma

    """
    nuc_ch = NUC_CH
    ###################
    # EXTRACT
    ###################
    struct_img = raw_soma_MCZ(in_img)

    nuclei = select_channel_from_raw(in_img, nuc_ch)
    nuclei = min_max_intensity_normalization(nuclei)

    ###################
    # PRE_PROCESSING
    ###################
    ################# part 1- soma
    struct_img = min_max_intensity_normalization(struct_img)

    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    # Linear-ish processing
    struct_img = median_filter_slice_by_slice(struct_img, size=median_sz_soma)

    struct_img = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gauss_sig_soma)

    struct_img_non_lin = non_linear_soma_transform_MCZ(struct_img)

    ################# part 2 - nuclei

    nuclei = median_filter_slice_by_slice(nuclei, size=median_sz_nuc)

    nuclei = image_smoothing_gaussian_slice_by_slice(nuclei, sigma=gauss_sig_nuc)

    ###################
    # CORE_PROCESSING
    ###################
    struct_obj = masked_object_thresh(
        struct_img_non_lin, th_method=mo_method, cutoff_size=mo_cutoff_size, th_adjust=mo_adjust
    )
    # # "Masked Object Thresholding" - 3D capable
    # struct_obj = MO(
    #     struct_img_non_lin,
    #     global_thresh_method=mo_method,
    #     object_minArea=mo_cutoff_size,
    #     local_adjust=mo_adjust,
    #     return_object=False
    # )
    ################# part 2 : nuclei thresholding
    nuclei_object = apply_log_li_threshold(
        nuclei, thresh_factor=thresh_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )

    # # wrapper to remoce_small_objects
    nuclei_object = hole_filling(nuclei_object, hole_min=0, hole_max=max_hole_w_nuc**2, fill_2d=True)

    nuclei_object = size_filter_2D(nuclei_object, min_size=small_obj_w_nuc**2, connectivity=1)

    nuclei_labels = label(nuclei_object)
    ###################
    # POST_PROCESSING
    ###################
    struct_obj = hole_filling(struct_obj, hole_min=0, hole_max=max_hole_w_soma**2, fill_2d=True)

    struct_obj = size_filter_2D(struct_obj, min_size=small_obj_w_soma**2, connectivity=1)

    labels_out = masked_inverted_watershed(
        struct_img, nuclei_labels, struct_obj
    )  # np.logical_or(struct_obj, NU_labels > 0)

    ###################
    # POST- POST_PROCESSING
    ###################
    # keep the "SOMA" label which contains the highest total signal
    soma_out = choose_max_label(struct_img, labels_out)

    return soma_out


##########################
# 1. fixed_infer_soma
##########################
def fixed_infer_soma(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer soma from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    Returns
    -------------
    soma_mask  -  a logical/labels object defining boundaries of soma
    """

    ###################
    # PARAMETERS
    ###################
    median_sz_soma = 15
    gauss_sig_soma = 1.34
    median_sz_nuc = 4
    gauss_sig_nuc = 1.34
    mo_method = "ave"
    mo_adjust = 0.5
    mo_cutoff_size = 100
    thresh_factor = 0.9
    thresh_min = 0.1
    thresh_max = 1.0
    max_hole_w_nuc = 5
    small_obj_w_nuc = 15
    max_hole_w_soma = 40
    small_obj_w_soma = 15

    soma_out = infer_soma(
        in_img,
        median_sz_soma,
        gauss_sig_soma,
        median_sz_nuc,
        gauss_sig_nuc,
        mo_method,
        mo_adjust,
        mo_cutoff_size,
        thresh_factor,
        thresh_min,
        thresh_max,
        max_hole_w_nuc,
        small_obj_w_nuc,
        max_hole_w_soma,
        small_obj_w_soma,
    )

    return soma_out

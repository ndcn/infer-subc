from skimage.filters import scharr
from skimage.measure import label

import numpy as np

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
    weighted_aggregate,
    masked_inverted_watershed,
    fill_and_filter_linear_size,
    get_max_label,
)


def _raw_soma_MCZ(img_in):
    """define soma image (DEPRICATED to leverage `weighted_aggregate`)
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


def raw_soma_MCZ(img_in: np.ndarray, scale_min_max: bool = True) -> np.ndarray:
    """define soma image
    SOMA_W = (6.,1.,2.)
    SOMA_CH = (LYSO_CH,ER_CH,GOLGI_CH)

    Parameters
    ------------
    img_in
        a 3d image
    scale_min_max:
        scale to [0,1] if True. default True

    Returns
    -------------
        np.ndarray scaled aggregate

    """
    weights = (0, 6, 0, 2, 0, 1)
    if scale_min_max:
        return min_max_intensity_normalization(weighted_aggregate(img_in, *weights))
    else:
        return weighted_aggregate(img_in, *weights)


def non_linear_soma_transform_MCZ(in_img):
    """non-linear distortion to fill out soma
    log + edge of smoothed composite

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    Returns
    -------------
        np.ndarray scaled aggregate
    """
    # non-Linear processing
    log_img, d = log_transform(in_img.copy())
    log_img = min_max_intensity_normalization(log_img)
    return min_max_intensity_normalization(scharr(log_img)) + log_img


def choose_max_label_soma_union_nucleus(
    soma_img: np.ndarray, soma_obj: np.ndarray, nuclei_obj: np.ndarray
) -> np.ndarray:
    """find the label with the maximum soma from watershed on the nuclei + plus the corresponding nuclei labels

        Parameters
    ------------
    soma_img:
        the soma image intensities
    soma_obj:
        thresholded soma mask
    nuclei_obj:
        inferred nuclei

    Returns
    -------------
        np.ndarray of soma+nuc labels corresponding to the largest total soma signal

    """
    nuc_labels = label(nuclei_obj)
    soma_labels = masked_inverted_watershed(soma_img, nuc_labels, soma_obj)

    keep_label = get_max_label(soma_img, soma_labels)

    soma_out = np.zeros_like(soma_labels)
    soma_out[soma_labels == keep_label] = 1
    soma_out[nuc_labels == keep_label] = 1

    return soma_out


##########################
# 1. infer_soma
##########################
# TODO:  break up the logic so the EXTRACT / PRE-PROCESS functions are more flexible? i.e. not nescessarily MCZ
def infer_soma_MCZ(
    in_img: np.ndarray,
    nuclei_obj: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    mo_method: str,
    mo_adjust: float,
    mo_cutoff_size: int,
    max_hole_w: int,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer soma from linearly unmixed input.

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    nuclei_obj:
        a 3d image containing the inferred nuclei
    median_sz:
        width of median filter for _soma_ signal
    gauss_sig:
        sigma for gaussian smoothing of _soma_ signal
    mo_method:
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
    mo_adjust:
        Masked Object threshold `local_adjust`
    mo_cutoff_size:
        Masked Object threshold `size_min`
    max_hole_w:
        hole filling cutoff for soma signal post-processing
    small_obj_w:
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
    scaled_signal = struct_img.copy()  # already scaled

    ###################
    # PRE_PROCESSING
    ###################
    ################# part 1- soma

    # Linear-ish processing
    struct_img = median_filter_slice_by_slice(struct_img, size=median_sz)

    struct_img = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gauss_sig)

    struct_img_non_lin = non_linear_soma_transform_MCZ(struct_img)

    ###################
    # CORE_PROCESSING
    ###################
    struct_obj = masked_object_thresh(
        struct_img_non_lin, th_method=mo_method, cutoff_size=mo_cutoff_size, th_adjust=mo_adjust
    )

    ###################
    # POST_PROCESSING
    ###################
    # struct_obj = hole_filling_linear_size(struct_obj,
    #                                             hole_min =0 ,
    #                                             hole_max=max_hole_w)
    # struct_obj = size_filter_linear_size(struct_obj,
    #                                                 min_size= small_obj_w)
    struct_obj = fill_and_filter_linear_size(struct_obj, hole_min=0, hole_max=max_hole_w, min_size=small_obj_w)

    ###################
    # POST- POST_PROCESSING
    ###################
    soma_out = choose_max_label_soma_union_nucleus(struct_img, struct_obj, nuclei_obj)

    return soma_out


def fixed_infer_soma_MCZ(in_img: np.ndarray, nuclei_obj: np.ndarray) -> np.ndarray:
    """
    Procedure to infer soma from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    nuclei_obj:
        a 3d image containing the inferred nuclei

    Returns
    -------------
    soma_mask:
        a logical/labels object defining boundaries of soma
    """

    ###################
    # PARAMETERS
    ###################
    median_sz = 15
    gauss_sig = 1.34
    mo_method = "ave"
    mo_adjust = 0.5
    mo_cutoff_size = 150
    max_hole_w = 50
    small_obj_w = 45

    soma_out = infer_soma_MCZ(
        in_img, nuclei_obj, median_sz, gauss_sig, mo_method, mo_adjust, mo_cutoff_size, max_hole_w, small_obj_w
    )

    return soma_out

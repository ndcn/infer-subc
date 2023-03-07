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
    size_filter_linear_size,
    choose_max_label,
    select_channel_from_raw,
    weighted_aggregate,
    masked_inverted_watershed,
    hole_filling_linear_size,
)


def _raw_soma_MCZ(img_in):
    """define soma image (DEPRICATED)
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


##########################
# 1. infer_soma
##########################
def infer_soma(
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
    struct_obj = hole_filling_linear_size(struct_obj, hole_min=0, hole_max=max_hole_w)

    struct_obj = size_filter_linear_size(struct_obj, min_size=small_obj_w)

    labels_out = masked_inverted_watershed(
        struct_img, label(nuclei_obj), struct_obj
    )  # np.logical_or(struct_obj, NU_labels > 0)

    ###################
    # POST- POST_PROCESSING
    ###################
    # keep the "SOMA" label which contains the highest total signal
    soma_out = choose_max_label(scaled_signal, labels_out)

    return soma_out


def fixed_infer_soma(in_img: np.ndarray, nuclei_obj: np.ndarray) -> np.ndarray:
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
    mo_cutoff_size = 100
    max_hole_w = 40
    small_obj_w = 15

    soma_out = infer_soma(
        in_img, nuclei_obj, median_sz, gauss_sig, mo_method, mo_adjust, mo_cutoff_size, max_hole_w, small_obj_w
    )

    return soma_out

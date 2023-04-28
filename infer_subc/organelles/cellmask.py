from typing import Dict
from pathlib import Path
import time
import numpy as np
from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice

from skimage.filters import scharr
from skimage.measure import label

from infer_subc.constants import (
    TEST_IMG_N,
    NUC_CH,
    LYSO_CH,
    MITO_CH,
    GOLGI_CH,
    PEROX_CH,
    ER_CH,
    LD_CH,
    RESIDUAL_CH,
)
from infer_subc.core.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc.core.img import (
    masked_object_thresh,
    log_transform,
    min_max_intensity_normalization,
    median_filter_slice_by_slice,
    scale_and_smooth,
    weighted_aggregate,
    masked_inverted_watershed,
    fill_and_filter_linear_size,
    get_max_label,
    get_interior_labels,
)


def raw_cellmask_fromaggr(img_in: np.ndarray, scale_min_max: bool = True) -> np.ndarray:
    """define cellmask image
    CELLMASK_W = (6.,1.,2.)
    CELLMASK_CH = (LYSO_CH,ER_CH,GOLGI_CH)

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


def non_linear_cellmask_transform_MCZ(in_img):
    """non-linear distortion to fill out cellmask
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


def choose_max_label_cellmask_union_nucleus(
    cellmask_img: np.ndarray, cellmask_obj: np.ndarray, nuclei_labels: np.ndarray, interior_labels: bool = True
) -> np.ndarray:
    """get cellmask UNION nuclei for largest signal label

        Parameters
    ------------
    cellmask_img:
        the cellmask image intensities
    cellmask_obj:
        thresholded cellmask mask
    nuclei_labels:
        inferred nuclei labels (np.uint16)

    Returns
    -------------
        boolean np.ndarray of cellmask+nuc corresponding to the label of largest total cellmask signal

    """

    cellmask_labels = masked_inverted_watershed(cellmask_img, nuclei_labels, cellmask_obj)

    # should we restrict to interior nuclear labels?
    # get_interior_labels(nuclei_object)
    # would need to update get_max_label to only choose the labels in get_interior_label
    target_labels = get_interior_labels(nuclei_labels) if interior_labels else None

    keep_label = get_max_label(cellmask_img, cellmask_labels, target_labels=target_labels)

    cellmask_out = np.zeros_like(cellmask_labels)
    cellmask_out[cellmask_labels == keep_label] = 1
    cellmask_out[nuclei_labels == keep_label] = 1

    return cellmask_out > 0


##########################
# 1. infer_cellmask
##########################
# TODO:  break up the logic so the EXTRACT / PRE-PROCESS functions are more flexible? i.e. not nescessarily MCZ
def infer_cellmask_fromaggr(
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
    Procedure to infer cellmask from linearly unmixed input.

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    nuclei_obj:
        a 3d image containing the inferred nuclei
    median_sz:
        width of median filter for _cellmask_ signal
    gauss_sig:
        sigma for gaussian smoothing of _cellmask_ signal
    mo_method:
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
    mo_adjust:
        Masked Object threshold `local_adjust`
    mo_cutoff_size:
        Masked Object threshold `size_min`
    max_hole_w:
        hole filling cutoff for cellmask signal post-processing
    small_obj_w:
        minimu object size cutoff for cellmask signal post-processing

    Returns
    -------------
    cellmask_mask:
        a logical/labels object defining boundaries of cellmask

    """
    # nuc_ch = NUC_CH
    ###################
    # EXTRACT
    ###################
    print(f"shape in_img {in_img.shape}")
    print(f"shape nuclei_obj {nuclei_obj.shape}")

    struct_img = raw_cellmask_fromaggr(in_img)
    # scaled_signal = struct_img.copy()  # already scaled

    ###################
    # PRE_PROCESSING
    ###################
    ################# part 1- cellmask
    print(f"shape struct_img {struct_img.shape}")

    # Linear-ish processing
    struct_img = scale_and_smooth(struct_img, median_sz=median_sz, gauss_sig=gauss_sig)

    struct_img_non_lin = non_linear_cellmask_transform_MCZ(struct_img)

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
    cellmask_out = choose_max_label_cellmask_union_nucleus(struct_img, struct_obj, nuclei_obj)

    return cellmask_out


def fixed_infer_cellmask_fromaggr(in_img: np.ndarray, nuclei_obj: np.ndarray) -> np.ndarray:
    """
    Procedure to infer cellmask from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    nuclei_obj:
        a 3d image containing the inferred nuclei

    Returns
    -------------
    cellmask_mask:
        a logical/labels object defining boundaries of cellmask
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

    cellmask_out = infer_cellmask_fromaggr(
        in_img, nuclei_obj, median_sz, gauss_sig, mo_method, mo_adjust, mo_cutoff_size, max_hole_w, small_obj_w
    )

    return cellmask_out


def infer_and_export_cellmask(
    in_img: np.ndarray, nuclei_obj: np.ndarray, meta_dict: Dict, out_data_path: Path
) -> np.ndarray:
    """
    infer cellmask and write inferred cellmask to ome.tif file

    Parameters
    ------------
    in_img:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    nuclei_obj:
        a 3d image containing the inferred nuclei
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """
    cellmask = fixed_infer_cellmask_fromaggr(in_img, nuclei_obj)
    out_file_n = export_inferred_organelle(cellmask, "cell", meta_dict, out_data_path)
    print(f"inferred cellmask. wrote {out_file_n}")
    return cellmask>0


def get_cellmask(in_img: np.ndarray, nuclei_obj: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    load cellmask if it exists, otherwise calculate and write inferred cellmask to ome.tif file

    Parameters
    ------------
    in_img:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    nuclei_obj:
        a 3d image containing the inferred nuclei
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """

    try:
        cellmask = import_inferred_organelle("cell", meta_dict, out_data_path)
    except:
        start = time.time()
        print("starting segmentation...")
        cellmask = fixed_infer_cellmask_fromaggr(in_img, nuclei_obj)
        out_file_n = export_inferred_organelle(cellmask, "cell", meta_dict, out_data_path)
        end = time.time()
        print(f"inferred (and exported) cellmask in ({(end - start):0.2f}) sec")

    return cellmask>0

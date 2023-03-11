import numpy as np
from typing import Dict
from pathlib import Path
import time
from infer_subc_2d.constants import MITO_CH
from infer_subc_2d.utils.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc_2d.utils.img import (
    size_filter_linear_size,
    size_filter_linear_size,
    vesselness_slice_by_slice,
    select_channel_from_raw,
    scale_and_smooth,
)

##########################
#  infer_mitochondria
##########################
def infer_mitochondria(
    in_img: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    vesselness_scale: float,
    vesselness_cut: float,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer mitochondria from linearly unmixed input.

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    median_sz:
        width of median filter for signal
    gauss_sig:
        sigma for gaussian smoothing of  signal
    vesselness_scale:
        scale (log_sigma) for vesselness filter
    vesselness_cut:
        threshold for vesselness fitered threshold
    small_obj_w:
        minimu object size cutoff for nuclei post-processing

    Returns
    -------------
    mitochondria_object
        mask defined extent of mitochondria
    """
    mito_ch = MITO_CH
    ###################
    # EXTRACT
    ###################
    mito = select_channel_from_raw(in_img, MITO_CH)

    ###################
    # PRE_PROCESSING
    ###################
    struct_img = scale_and_smooth(mito, median_sz=median_sz, gauss_sig=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    struct_img = vesselness_slice_by_slice(struct_img, sigma=vesselness_scale, cutoff=vesselness_cut, tau=0.75)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = size_filter_linear_size(struct_img, min_size=small_obj_w)

    return struct_obj


##########################
#  fixed_infer_mitochondria
##########################
def fixed_infer_mitochondria(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer mitochondria from linearly unmixed input

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    Returns
    -------------
    mitochondria_object
        mask defined extent of mitochondria
    """
    median_sz = 3
    gauss_sig = 1.4
    vesselness_scale = 1.5
    vesselness_cut = 0.05
    small_obj_w = 3

    return infer_mitochondria(in_img, median_sz, gauss_sig, vesselness_scale, vesselness_cut, small_obj_w)


def infer_and_export_mitochondria(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    infer mitochondria and write inferred mitochondria to ome.tif file

    Parameters
    ------------
    in_img:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """
    mitochondria = fixed_infer_mitochondria(in_img)
    out_file_n = export_inferred_organelle(mitochondria, "mitochondria", meta_dict, out_data_path)
    print(f"inferred mitochondria. wrote {out_file_n}")
    return mitochondria


def get_mitochondria(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    load mitochondria if it exists, otherwise calculate and write to ome.tif file

    Parameters
    ------------
    in_img:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """

    try:
        mitochondria = import_inferred_organelle("mitochondria", meta_dict, out_data_path)
    except:
        start = time.time()
        print("starting segmentation...")
        mitochondria = infer_and_export_mitochondria(in_img, meta_dict, out_data_path)
        end = time.time()
        print(f"inferred (and exported) mitochondria in ({(end - start):0.2f}) sec")

    return mitochondria

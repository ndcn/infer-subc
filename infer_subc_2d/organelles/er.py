import numpy as np
from typing import Dict
from pathlib import Path
import time

from infer_subc_2d.constants import ER_CH
from infer_subc_2d.core.file_io import export_inferred_organelle, import_inferred_organelle

from infer_subc_2d.core.img import (
    size_filter_linear_size,
    select_channel_from_raw,
    filament_filter,
    # normalized_edge_preserving_smoothing,
    scale_and_smooth,
    label_uint16,
)


##########################
#  infer_ER
##########################
def infer_ER(
    in_img: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    filament_scale: float,
    filament_cut: float,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer peroxisome from linearly unmixed input.

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    median_sz:
        width of median filter for signal
    gauss_sig:
        sigma for gaussian smoothing of  signal
    filament_scale:
        scale (log_sigma) for filament filter
    filament_cut:
        threshold for filament fitered threshold
    small_obj_w:
        minimu object size cutoff for nuclei post-processing
    Returns
    -------------
    peroxi_object
        mask defined extent of peroxisome object
    """
    er_ch = ER_CH
    ###################
    # EXTRACT
    ###################
    er = select_channel_from_raw(in_img, er_ch)

    ###################
    # PRE_PROCESSING
    ###################
    # er = normalized_edge_preserving_smoothing(er)
    struct_img = scale_and_smooth(er, median_sz=median_sz, gauss_sig=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    # f2_param = [[filament_scale, filament_cut]]
    # # f2_param = [[1, 0.15]]  # [scale_1, cutoff_1]
    # struct_obj = filament_2d_wrapper(er, f2_param)
    struct_obj = filament_filter(struct_img, filament_scale, filament_cut)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = size_filter_linear_size(struct_obj, min_size=small_obj_w)

    return label_uint16(struct_obj)


##########################
#  fixed_infer_ER
##########################
def fixed_infer_ER(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer endoplasmic rediculum from linearly unmixed input with *fixed parameters*

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    Returns
    -------------
    peroxi_object
        mask defined extent of peroxisome object
    """
    median_sz = 3
    gauss_sig = 2.0
    filament_scale = 1
    filament_cut = 0.015
    small_obj_w = 2
    return infer_ER(in_img, median_sz, gauss_sig, filament_scale, filament_cut, small_obj_w)


def infer_and_export_ER(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    infer ER and write inferred ER to ome.tif file

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
    er = fixed_infer_ER(in_img)
    out_file_n = export_inferred_organelle(er, "er", meta_dict, out_data_path)
    print(f"inferred ER. wrote {out_file_n}")
    return er


def get_ER(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    load endoplasmic_reticulum if it exists, otherwise calculate and write to ome.tif file

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
        er = import_inferred_organelle("er", meta_dict, out_data_path)
    except:
        start = time.time()
        print("starting segmentation...")
        er = infer_and_export_ER(in_img, meta_dict, out_data_path)
        end = time.time()
        print(f"inferred (and exported) er in ({(end - start):0.2f}) sec")

    return er

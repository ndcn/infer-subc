import numpy as np
from typing import Dict
from pathlib import Path
import time

from infer_subc_2d.constants import ER_CH
from infer_subc_2d.utils.file_io import export_inferred_organelle, import_inferred_organelle

from infer_subc_2d.utils.img import (
    size_filter_linear_size,
    select_channel_from_raw,
    filament_filter,
    normalized_edge_preserving_smoothing,
)

##########################
#  infer_endoplasmic_reticulum
##########################
def infer_endoplasmic_reticulum(
    in_img: np.ndarray,
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
    er = normalized_edge_preserving_smoothing(er)

    ###################
    # CORE_PROCESSING
    ###################
    # f2_param = [[filament_scale, filament_cut]]
    # # f2_param = [[1, 0.15]]  # [scale_1, cutoff_1]
    # struct_obj = filament_2d_wrapper(er, f2_param)
    struct_obj = filament_filter(er, filament_scale, filament_cut)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = size_filter_linear_size(struct_obj, min_size=small_obj_w)

    return struct_obj


##########################
#  fixed_infer_endoplasmic_reticulum
##########################
def fixed_infer_endoplasmic_reticulum(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer endoplasmic rediculum from linearly unmixed input with *fixed parameters*

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    cytosol_mask:
        mask - default=None

    Returns
    -------------
    peroxi_object
        mask defined extent of er object
    """
    filament_scale = 1
    filament_cut = 0.15
    small_obj_w = 2
    return infer_endoplasmic_reticulum(in_img, filament_scale, filament_cut, small_obj_w)


def infer_and_export_endoplasmic_reticulum(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    infer endoplasmic reticulum and write inferred endoplasmic reticulum to ome.tif file

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
    er = fixed_infer_endoplasmic_reticulum(in_img)
    out_file_n = export_inferred_organelle(er, "er", meta_dict, out_data_path)
    print(f"inferred endoplasmic reticulum. wrote {out_file_n}")
    return er

def get_endoplasmic_reticulum(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
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
        er = infer_and_export_endoplasmic_reticulum(in_img, meta_dict, out_data_path)
        end = time.time()
        print(f"inferred (and exported) er in ({(end - start):0.2f}) sec")

    return er

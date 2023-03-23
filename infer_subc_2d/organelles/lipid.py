import numpy as np
from typing import Dict
from pathlib import Path
import time

from infer_subc_2d.core.img import (
    apply_threshold,
    scale_and_smooth,
    fill_and_filter_linear_size,
    select_channel_from_raw,
)
from infer_subc_2d.core.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc_2d.constants import LD_CH


##########################
#  infer_LD
##########################
def infer_LD(
    in_img: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    method: str,
    thresh_factor: float,
    thresh_min: float,
    thresh_max: float,
    max_hole_w: int,
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
    method:
        method for applying threshold.  "otsu"  or "li", "triangle", "median", "ave", "sauvola","multi_otsu","muiltiotsu"
    thresh_factor:
        scaling value for threshold
    thresh_min:
        absolute minumum for threshold
    thresh_max:
        absolute maximum for threshold
    max_hole_w:
        hole filling cutoff for lipid post-processing
    small_obj_w:
        minimu object size cutoff for lipid post-processing
    Returns
    -------------
    peroxi_object
        mask defined extent of peroxisome object
    """
    LD_ch = LD_CH
    ###################
    # EXTRACT
    ###################
    lipid = select_channel_from_raw(in_img, LD_ch)
    ###################
    # PRE_PROCESSING
    ###################
    lipid = scale_and_smooth(lipid, median_sz=median_sz, gauss_sig=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    bw = apply_threshold(
        lipid, method=method, thresh_factor=thresh_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )

    ###################
    # POST_PROCESSING
    ###################
    # min_hole_w = 0
    struct_obj = fill_and_filter_linear_size(bw, hole_min=0, hole_max=max_hole_w, min_size=small_obj_w)

    return struct_obj


##########################
#  fixed_infer_LD
##########################
def fixed_infer_LD(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer cellmask from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    Returns
    -------------
    LD_body_object
        mask defined extent of liipid body

    """
    median_sz = 2
    gauss_sig = 1.34
    method = "otsu"
    threshold_factor = 0.99  # from cellProfiler
    thresh_min = 0.5
    thresh_max = 1.0
    max_hole_w = 2.5
    small_obj_w = 4

    return infer_LD(
        in_img, median_sz, gauss_sig, method, threshold_factor, thresh_min, thresh_max, max_hole_w, small_obj_w
    )


def infer_and_export_LD(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    infer lipid bodies and write inferred lipid to ome.tif file

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
    lipid = fixed_infer_LD(in_img)
    out_file_n = export_inferred_organelle(lipid, "lipid", meta_dict, out_data_path)
    print(f"inferred lipid. wrote {out_file_n}")
    return lipid


def get_LD(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    load lipid if it exists, otherwise calculate and write to ome.tif file

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
        lipid = import_inferred_organelle("lipid", meta_dict, out_data_path)
    except:
        start = time.time()
        print("starting segmentation...")
        lipid = infer_and_export_LD(in_img, meta_dict, out_data_path)
        end = time.time()
        print(f"inferred (and exported) lipid in ({(end - start):0.2f}) sec")

    return lipid

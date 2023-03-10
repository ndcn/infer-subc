import numpy as np
from typing import Dict
from pathlib import Path

from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.vessel import filament_2d_wrapper

from infer_subc_2d.constants import LYSO_CH
from infer_subc_2d.utils.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc_2d.utils.img import (
    scale_and_smooth,
    fill_and_filter_linear_size,
    select_channel_from_raw,
)

##########################
#  infer_LYSOSOMES
##########################
def infer_lysosome(
    in_img: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    dot_scale_1: float,
    dot_cut_1: float,
    dot_scale_2: float,
    dot_cut_2: float,
    dot_scale_3: float,
    dot_cut_3: float,
    filament_scale: float,
    filament_cut: float,
    min_hole_w: int,
    max_hole_w: int,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer lysosome from linearly unmixed input

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    median_sz:
        width of median filter for signal
    gauss_sig:
        sigma for gaussian smoothing of  signal
    dot_scale:
        scales (log_sigma) for dot filter (1,2, and 3)
    dot_cut:
        threshold for dot filter thresholds (1,2,and 3)
    filament_scale:
        scale (log_sigma) for filament filter
    filament_cut:
        threshold for filament fitered threshold
    min_hole_w:
        hole filling min for nuclei post-processing
    max_hole_w:
        hole filling cutoff for nuclei post-processing
    small_obj_w:
        minimu object size cutoff for nuclei post-processing

    Returns
    -------------
    lysosome_object:
        mask defined extent of lysosome object

    """
    lyso_ch = LYSO_CH
    ###################
    # EXTRACT
    ###################
    lyso = select_channel_from_raw(in_img, lyso_ch)

    ###################
    # PRE_PROCESSING
    ###################
    lyso = scale_and_smooth(lyso, median_sz=median_sz, gauss_sig=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    s2_param = [[dot_scale_1, dot_cut_1], [dot_scale_2, dot_cut_2], [dot_scale_3, dot_cut_3]]
    bw_spot = dot_2d_slice_by_slice_wrapper(lyso, s2_param)

    f2_param = [[filament_scale, filament_cut]]
    bw_filament = filament_2d_wrapper(lyso, f2_param)
    # TODO: consider 3D version to call: aicssegmentation::vesselness3D

    bw = np.logical_or(bw_spot, bw_filament)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = fill_and_filter_linear_size(bw, hole_min=min_hole_w, hole_max=max_hole_w, min_size=small_obj_w)
    return struct_obj


##########################
#  fixed_infer_nuclei
##########################
def fixed_infer_lysosome(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer lysosome from linearly unmixed input

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    Returns
    -------------
    nuclei_object:
        mask defined extent of NU
    """
    median_sz = 4
    gauss_sig = 1.34
    dot_scale_1 = 5
    dot_cut_1 = 0.09
    dot_scale_2 = 2.5
    dot_cut_2 = 0.07
    dot_scale_3 = 1
    dot_cut_3 = 0.01
    filament_scale = 1
    filament_cut = 0.15
    min_hole_w = 0
    max_hole_w = 25
    small_obj_w = 3

    return infer_lysosome(
        in_img,
        median_sz,
        gauss_sig,
        dot_cut_1,
        dot_scale_1,
        dot_cut_2,
        dot_scale_2,
        dot_cut_3,
        dot_scale_3,
        filament_scale,
        filament_cut,
        min_hole_w,
        max_hole_w,
        small_obj_w,
    )


def infer_and_export_lysosome(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    infer lysosome and write inferred lysosome to ome.tif file

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
    lysosome = fixed_infer_lysosome(in_img)
    out_file_n = export_inferred_organelle(lysosome, "lysosome", meta_dict, out_data_path)
    print(f"inferred lysosome. wrote {out_file_n}")
    return lysosome


def lysosome_spot_filter(in_img: np.ndarray) -> np.ndarray:
    """spot filter helper function for lysosome"""
    dot_scale_1 = 5
    dot_cut_1 = 0.09
    dot_scale_2 = 2.5
    dot_cut_2 = 0.07
    dot_scale_3 = 1
    dot_cut_3 = 0.01
    s2_param = [[dot_scale_1, dot_cut_1], [dot_scale_2, dot_cut_2], [dot_scale_3, dot_cut_3]]
    return dot_2d_slice_by_slice_wrapper(in_img, s2_param)


def lysosome_filiment_filter(in_img: np.ndarray) -> np.ndarray:
    """spot filter helper function for lysosome (DEPRICATED)"""
    filament_scale = 1
    filament_cut = 0.15
    f2_param = [[filament_scale, filament_cut]]
    # f2_param = [[1, 0.15]]  # [scale_1, cutoff_1]
    return filament_2d_wrapper(in_img, f2_param)


def get_lysosome(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    load lysosome if it exists, otherwise calculate and write to ome.tif file

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

    lysosome = import_inferred_organelle("lysosome", meta_dict, out_data_path)

    if lysosome is None:
        lysosome = infer_and_export_lysosome(in_img, meta_dict, out_data_path)
    else:
        print(f"loaded lysosome from {out_data_path}")

    return lysosome

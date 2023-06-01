import numpy as np
from typing import Dict
from pathlib import Path
import time

from infer_subc.constants import ER_CH
from infer_subc.core.file_io import export_inferred_organelle, import_inferred_organelle

from infer_subc.core.img import (
    select_channel_from_raw,
    scale_and_smooth,
    label_bool_as_uint16,
    fill_and_filter_linear_size,
    masked_object_thresh,
    filament_filter_3
)


##########################
#  infer_ER
##########################
def infer_ER(
              in_img: np.ndarray,
              ER_ch: int,
              median_sz: int,
              gauss_sig: float,
              MO_thresh_method: str,
              MO_cutoff_size: float,
              MO_thresh_adj: float,
              fil_scale_1: float,
              fil_cut_1: float,
              fil_scale_2: float, 
              fil_cut_2: float, 
              fil_scale_3: float, 
              fil_cut_3: float,
              fil_method: str,
              min_hole_w: int,
              max_hole_w: int,
              small_obj_w: int,
              fill_filter_method: str
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

    ###################
    # EXTRACT
    ###################    
    ER = select_channel_from_raw(in_img, ER_ch)

    ###################
    # PRE_PROCESSING
    ###################    
    # er = normalized_edge_preserving_smoothing(er)
    struct_img =  scale_and_smooth(ER,
                                   median_size = median_sz, 
                                   gauss_sigma = gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    bw1 = masked_object_thresh(struct_img, 
                                    global_method=MO_thresh_method, 
                                    cutoff_size=MO_cutoff_size, 
                                    local_adjust=MO_thresh_adj)

    bw2 = filament_filter_3(struct_img, fil_scale_1, fil_cut_1, fil_scale_2, fil_cut_2, fil_scale_3, fil_cut_3, fil_method)

    struct_obj = np.logical_or(bw1, bw2)
    
    ###################
    # POST_PROCESSING
    ################### 
    struct_obj = fill_and_filter_linear_size(struct_obj, 
                                             hole_min=min_hole_w, 
                                             hole_max=max_hole_w, 
                                             min_size=small_obj_w,
                                             method=fill_filter_method)

    ###################
    # LABELING
    ###################
    
    # ENSURE THAT there is ONLY ONE ER
    struct_obj = label_bool_as_uint16(struct_obj)

    return struct_obj 

# def infer_ER(
#     in_img: np.ndarray,
#     median_sz: int,
#     gauss_sig: float,
#     filament_scale: float,
#     filament_cut: float,
#     small_obj_w: int,
# ) -> np.ndarray:
#     """
#     Procedure to infer peroxisome from linearly unmixed input.

#     Parameters
#     ------------
#     in_img:
#         a 3d image containing all the channels
#     median_sz:
#         width of median filter for signal
#     gauss_sig:
#         sigma for gaussian smoothing of  signal
#     filament_scale:
#         scale (log_sigma) for filament filter
#     filament_cut:
#         threshold for filament fitered threshold
#     small_obj_w:
#         minimu object size cutoff for nuclei post-processing
#     Returns
#     -------------
#     peroxi_object
#         mask defined extent of peroxisome object
#     """
#     er_ch = ER_CH
#     ###################
#     # EXTRACT
#     ###################
#     er = select_channel_from_raw(in_img, er_ch)

#     ###################
#     # PRE_PROCESSING
#     ###################
#     # er = normalized_edge_preserving_smoothing(er)
#     struct_img = scale_and_smooth(er, median_sz=median_sz, gauss_sig=gauss_sig)

#     ###################
#     # CORE_PROCESSING
#     ###################
#     # f2_param = [[filament_scale, filament_cut]]
#     # # f2_param = [[1, 0.15]]  # [scale_1, cutoff_1]
#     # struct_obj = filament_2d_wrapper(er, f2_param)
#     struct_obj = filament_filter(struct_img, filament_scale, filament_cut)

#     ###################
#     # POST_PROCESSING
#     ###################
#     struct_obj = size_filter_linear_size(struct_obj, min_size=small_obj_w)

#     return label_bool_as_uint16(struct_obj)


##########################
#  fixed_infer_ER
##########################
def fixed_infer_ER(in_img: np.ndarray ) -> np.ndarray:
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
    ER_ch = 5
    median_sz = 3
    gauss_sig = 2.0
    MO_thresh_method = 'tri'
    MO_cutoff_size = 1200
    MO_thresh_adj = 0.7
    fil_scale_1 = 1
    fil_cut_1 = 0.005
    fil_scale_2 = 2
    fil_cut_2 = 0.005
    fil_scale_3 = 0
    fil_cut_3 = 0
    fil_method = "3D"
    min_hole_w = 0
    max_hole_w = 0
    small_obj_w = 4
    method = "3D"

    return infer_ER(
        in_img,
        ER_ch,
        median_sz,
        gauss_sig,
        MO_thresh_method,
        MO_cutoff_size,
        MO_thresh_adj,
        fil_scale_1,
        fil_cut_1,
        fil_scale_2,
        fil_cut_2,
        fil_scale_3,
        fil_cut_3,
        fil_method,
        min_hole_w,
        max_hole_w,
        small_obj_w,
        method)

#  def fixed_infer_ER(in_img: np.ndarray) -> np.ndarray:
#     """
#     Procedure to infer endoplasmic rediculum from linearly unmixed input with *fixed parameters*

#     Parameters
#     ------------
#     in_img:
#         a 3d image containing all the channels

#     Returns
#     -------------
#     peroxi_object
#         mask defined extent of peroxisome object
#     """
#     median_sz = 3
#     gauss_sig = 2.0
#     filament_scale = 1
#     filament_cut = 0.015
#     small_obj_w = 2
#     return infer_ER(in_img, median_sz, gauss_sig, filament_scale, filament_cut, small_obj_w)


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
    out_file_n = export_inferred_organelle(er, "ER", meta_dict, out_data_path)
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
        er = import_inferred_organelle("ER", meta_dict, out_data_path)
    except:
        start = time.time()
        print("starting segmentation...")
        er = infer_and_export_ER(in_img, meta_dict, out_data_path)
        end = time.time()
        print(f"inferred (and exported) ER in ({(end - start):0.2f}) sec")

    return er

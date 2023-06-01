import numpy as np
from typing import Dict
from pathlib import Path
import time
from infer_subc.constants import MITO_CH
from infer_subc.core.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc.core.img import (
    select_channel_from_raw,
    scale_and_smooth,
    fill_and_filter_linear_size,
    label_uint16,
    dot_filter_3,
    filament_filter_3
)


##########################
#  infer_mito
##########################
def infer_mito(
                in_img: np.ndarray,
                mito_ch: int,
                median_sz: int,
                gauss_sig: float,
                dot_scale_1: float,
                dot_cut_1: float,
                dot_scale_2: float,
                dot_cut_2: float,
                dot_scale_3: float,
                dot_cut_3: float,
                dot_method: str,
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
    mito_object
        mask defined extent of mitochondria object
    
    """

    ###################
    # EXTRACT
    ###################    
    mito = select_channel_from_raw(in_img, mito_ch)
    
    ###################
    # PRE_PROCESSING
    ###################                         
    mito =  scale_and_smooth(mito,
                             median_size = median_sz, 
                             gauss_sigma = gauss_sig)
    ###################   
    # CORE_PROCESSING
    ###################
    bw_dot = dot_filter_3(mito, dot_scale_1, dot_cut_1, dot_scale_2, dot_cut_2, dot_scale_3, dot_cut_3, dot_method)

    bw_filament = filament_filter_3(mito, fil_scale_1, fil_cut_1, fil_scale_2, fil_cut_2, fil_scale_3, fil_cut_3, fil_method)

    bw = np.logical_or(bw_dot, bw_filament)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = fill_and_filter_linear_size(bw, 
                                             hole_min=min_hole_w, 
                                             hole_max=max_hole_w, 
                                             min_size=small_obj_w,
                                             method=fill_filter_method)

    ###################
    # LABELING
    ###################
    struct_obj1 = label_uint16(struct_obj)

    return struct_obj1

# def infer_mito(
#     in_img: np.ndarray,
#     median_sz: int,
#     gauss_sig: float,
#     vesselness_scale: float,
#     vesselness_cut: float,
#     small_obj_w: int,
# ) -> np.ndarray:
#     """
#     Procedure to infer mitochondria from linearly unmixed input.

#     Parameters
#     ------------
#     in_img:
#         a 3d image containing all the channels
#     median_sz:
#         width of median filter for signal
#     gauss_sig:
#         sigma for gaussian smoothing of  signal
#     vesselness_scale:
#         scale (log_sigma) for vesselness filter
#     vesselness_cut:
#         threshold for vesselness fitered threshold
#     small_obj_w:
#         minimu object size cutoff for nuclei post-processing

#     Returns
#     -------------
#     mito_object
#         mask defined extent of mitochondria
#     """
#     mito_ch = MITO_CH
#     ###################
#     # EXTRACT
#     ###################
#     mito = select_channel_from_raw(in_img, MITO_CH)

#     ###################
#     # PRE_PROCESSING
#     ###################
#     struct_img = scale_and_smooth(mito, median_sz=median_sz, gauss_sig=gauss_sig)

#     ###################
#     # CORE_PROCESSING
#     ###################
#     struct_img = vesselness_slice_by_slice(struct_img, sigma=vesselness_scale, cutoff=vesselness_cut, tau=0.75)

#     ###################
#     # POST_PROCESSING
#     ###################
#     struct_obj = size_filter_linear_size(struct_img, min_size=small_obj_w)

#     return label_uint16(struct_obj)


##########################
#  fixed_infer_mito
##########################
def fixed_infer_mito(in_img: np.ndarray ) -> np.ndarray:
    """
    Procedure to infer mitochondria from linearly unmixed input,
    
    Parameters
    ------------
    in_img: 
        a 3d image containing all the channels

    Returns
    -------------
    mito_object
        mask defined extent of mitochondria
    """
    mito_ch = 2
    median_sz = 3
    gauss_sig = 1.34
    dot_scale_1 = 1.5
    dot_cut_1 = 0.05
    dot_scale_2 = 0
    dot_cut_2 = 0
    dot_scale_3 = 0
    dot_cut_3 = 0
    dot_method = "3D"
    fil_scale_1 = 1
    fil_cut_1 = 0.15
    fil_scale_2 = 0
    fil_cut_2 = 0
    fil_scale_3 = 0
    fil_cut_3 = 0
    fil_method = "3D"
    min_hole_w = 0
    max_hole_w = 0
    small_obj_w = 3
    method = "3D"

    return infer_mito(  
        in_img,
        mito_ch,
        median_sz,
        gauss_sig,
        dot_scale_1,
        dot_cut_1,
        dot_scale_2,
        dot_cut_2,
        dot_scale_3,
        dot_cut_3,
        dot_method,
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


# def fixed_infer_mito(in_img: np.ndarray) -> np.ndarray:
#     """
#     Procedure to infer mitochondria from linearly unmixed input

#     Parameters
#     ------------
#     in_img:
#         a 3d image containing all the channels

#     Returns
#     -------------
#     mito_object
#         mask defined extent of mitochondria
#     """
#     median_sz = 3
#     gauss_sig = 1.4
#     vesselness_scale = 1.5
#     vesselness_cut = 0.05
#     small_obj_w = 3

#     return infer_mito(in_img, median_sz, gauss_sig, vesselness_scale, vesselness_cut, small_obj_w)


def infer_and_export_mito(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
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
    mitochondria = fixed_infer_mito(in_img)
    out_file_n = export_inferred_organelle(mitochondria, "mito", meta_dict, out_data_path)
    print(f"inferred mitochondria. wrote {out_file_n}")
    return mitochondria


def get_mito(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
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
        mitochondria = import_inferred_organelle("mito", meta_dict, out_data_path)
    except:
        start = time.time()
        print("starting segmentation...")
        mitochondria = infer_and_export_mito(in_img, meta_dict, out_data_path)
        end = time.time()
        print(f"inferred (and exported) mitochondria in ({(end - start):0.2f}) sec")

    return mitochondria

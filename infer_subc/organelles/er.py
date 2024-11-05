import numpy as np
from typing import Dict
from pathlib import Path
import time

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
def infer_ER(in_img: np.ndarray,
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
    Procedure to infer ER from linearly unmixed input.

    Parameters
    ------------
    in_img: 
        a 3d image containing all the channels (CZYX)
    ER_ch:
        index of the ER channel in the input image
    median_sz: 
        width of median filter for signal
    gauss_sig: 
        sigma for gaussian smoothing of  signal
    mo_thresh_method: 
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
    mo_cutoff_size: 
        Masked Object threshold `size_min`
    mo_thresh_adjust: 
        Masked Object threshold `local_adjust`
    fil_scale_1: 
        scale (log_sigma) for filament filter
    fil_cutoff_1: 
        threshold for filament fitered threshold, associated to fil_scale_1
    fil_scale_2: 
        scale (log_sigma) for filament filter
    fil_cutoff_2: 
        threshold for filament fitered threshold, associated to fil_scale_2
    fil_scale_3: 
        scale (log_sigma) for filament filter
    fil_cutoff_3: 
        threshold for filament fitered threshold, associated to fil_scale_3
    fil_method:
        decision to process the filaments "slice-by-slice" or in "3D"
    min_hole_w: 
        minimum size for hole filling for cellmask signal post-processing
    max_hole_w: 
        hole filling cutoff for ER signal post-processing
    small_obj_w: 
        minimum object size cutoff for ER signal post-processing
    fill_filter_method:
        determines if small hole filling and small object removal should be run 'sice-by-slice' or in '3D'
    Returns
    -------------
    ER_object
        mask defined extent of ER object
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










#################################################################
########################## DEPRICATING ##########################
#################################################################


# ##########################
# #  fixed_infer_ER
# ##########################
# def fixed_infer_ER(in_img: np.ndarray ) -> np.ndarray:
#     """
#     Procedure to infer endoplasmic rediculum from linearly unmixed input with *fixed parameters*

#     Parameters
#     ------------
#     in_img: 
#         a 3d image containing all the channels

#     Returns
#     -------------
#     ER_object
#         mask defined extent of ER object
#     """
#     ER_ch = 5
#     median_sz = 3
#     gauss_sig = 2.0
#     MO_thresh_method = 'tri'
#     MO_cutoff_size = 1200
#     MO_thresh_adj = 0.7
#     fil_scale_1 = 1
#     fil_cut_1 = 0.005
#     fil_scale_2 = 2
#     fil_cut_2 = 0.005
#     fil_scale_3 = 0
#     fil_cut_3 = 0
#     fil_method = "3D"
#     min_hole_w = 0
#     max_hole_w = 0
#     small_obj_w = 4
#     method = "3D"

#     return infer_ER(
#         in_img,
#         ER_ch,
#         median_sz,
#         gauss_sig,
#         MO_thresh_method,
#         MO_cutoff_size,
#         MO_thresh_adj,
#         fil_scale_1,
#         fil_cut_1,
#         fil_scale_2,
#         fil_cut_2,
#         fil_scale_3,
#         fil_cut_3,
#         fil_method,
#         min_hole_w,
#         max_hole_w,
#         small_obj_w,
#         method)


# def infer_and_export_ER(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
#     """
#     infer ER and write inferred ER to ome.tif file

#     Parameters
#     ------------
#     in_img:
#         a 3d  np.ndarray image of the inferred organelle (labels or boolean)
#     meta_dict:
#         dictionary of meta-data (ome)
#     out_data_path:
#         Path object where tiffs are written to

#     Returns
#     -------------
#     exported file name

#     """
#     er = fixed_infer_ER(in_img)
#     out_file_n = export_inferred_organelle(er, "ER", meta_dict, out_data_path)
#     print(f"inferred ER. wrote {out_file_n}")
#     return er


# def get_ER(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
#     """
#     load endoplasmic_reticulum if it exists, otherwise calculate and write to ome.tif file

#     Parameters
#     ------------
#     in_img:
#         a 3d  np.ndarray image of the inferred organelle (labels or boolean)
#     meta_dict:
#         dictionary of meta-data (ome)
#     out_data_path:
#         Path object where tiffs are written to

#     Returns
#     -------------
#     exported file name

#     """

#     try:
#         er = import_inferred_organelle("ER", meta_dict, out_data_path)
#     except:
#         start = time.time()
#         print("starting segmentation...")
#         er = infer_and_export_ER(in_img, meta_dict, out_data_path)
#         end = time.time()
#         print(f"inferred (and exported) ER in ({(end - start):0.2f}) sec")

#     return er

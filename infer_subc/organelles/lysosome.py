import numpy as np
from typing import Dict
from pathlib import Path
import time

from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.vessel import filament_2d_wrapper

from infer_subc.core.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc.core.img import scale_and_smooth, fill_and_filter_linear_size, select_channel_from_raw, label_uint16, dot_filter_3, filament_filter_3


##########################
#  infer_LYSOSOMES
##########################
### USED ###
def infer_lyso(
                                in_img: np.ndarray,
                                lyso_ch: int,
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
    Procedure to infer lysosome from linearly unmixed input,
    
    Parameters
    ------------
    in_img: 
        a 3d image containing all the channels
    lyso_ch:
        index of the lyso channel in the input image (CZYX)
    median_sz: 
        width of median filter for signal
    gauss_sig: 
        sigma for gaussian smoothing of  signal
    dot_scale_1: 
        scales (log_sigma) for dot filter 1
    dot_cutoff_1: 
        threshold for dot filter thresholds associated to dot_scale_1
    dot_scale_2: 
        scales (log_sigma) for dot filter 1
    dot_cutoff_2: 
        threshold for dot filter thresholds associated to dot_scale_2
    dot_scale_3: 
        scales (log_sigma) for dot filter 1
    dot_cutoff_3: 
        threshold for dot filter thresholds associated to dot_scale_3
    dot_method:
        decision to process the dots "slice-by-slice" or in "3D"
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
        hole filling min for lyso post-processing
    max_hole_w: 
        hole filling cutoff for lyso post-processing
    small_obj_w: 
        minimum object size cutoff for lyso post-processing
    fill_filter_method:
        to fill small holes and remove small objects in "3D" or "slice-by-slice"

    Returns
    -------------
    lyso_object
        mask defined extent of lysosome object

    """
    ###################
    # EXTRACT
    ###################    
    lyso = select_channel_from_raw(in_img, lyso_ch)

     ###################
    # PRE_PROCESSING
    ###################    
    lyso1 =  scale_and_smooth(lyso,
                             median_size = median_sz, 
                             gauss_sigma = gauss_sig)
   ###################
    # CORE_PROCESSING
    ###################
    bw_dot = dot_filter_3(lyso1, dot_scale_1, dot_cut_1, dot_scale_2, dot_cut_2, dot_scale_3, dot_cut_3, dot_method)

    bw_filament = filament_filter_3(lyso1, fil_scale_1, fil_cut_1, fil_scale_2, fil_cut_2, fil_scale_3, fil_cut_3, fil_method)

    bw = np.logical_or(bw_dot, bw_filament)

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = fill_and_filter_linear_size(bw, hole_min=min_hole_w, hole_max=max_hole_w, min_size=small_obj_w, method=fill_filter_method)

    ###################
    # LABELING
    ###################
    struct_obj1 = label_uint16(struct_obj)

    return struct_obj1










#################################################################
########################## DEPRICATING ##########################
#################################################################

##########################
#  fixed_infer_nuclei
##########################

# def fixed_infer_lyso(in_img: np.ndarray) -> np.ndarray:
#     """
#     Procedure to infer lysosome from linearly unmixed input with *fixed parameters*
#     Parameters
#     ------------
#     in_img: 
#         a 3d image containing all the channels

#     Returns
#     -------------
#     lyso_object
#         mask defined extent of LS
#     """
#     lyso_ch = 1
#     median_sz = 3
#     gauss_sig = 1.34
#     dot_scale_1 = 5
#     dot_cut_1 = 0.09
#     dot_scale_2 = 2.5
#     dot_cut_2 = 0.07
#     dot_scale_3 = 1
#     dot_cut_3 = 0.01
#     dot_method = "3D"
#     fil_scale_1 = 1
#     fil_cut_1 = 0.15
#     fil_scale_2 = 0
#     fil_cut_2 = 0
#     fil_scale_3 = 0
#     fil_cut_3 = 0
#     fil_method = "3D"
#     min_hole_w = 0
#     max_hole_w = 25
#     small_obj_w = 0
#     method = "3D"

#     return infer_lyso(  
#         in_img,
#         lyso_ch,
#         median_sz,
#         gauss_sig,
#         dot_scale_1,
#         dot_cut_1,
#         dot_scale_2,
#         dot_cut_2,
#         dot_scale_3,
#         dot_cut_3,
#         dot_method,
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


# def fixed_infer_lyso(in_img: np.ndarray) -> np.ndarray:
#     """
#     Procedure to infer lyso from linearly unmixed input

#     Parameters
#     ------------
#     in_img:
#         a 3d image containing all the channels

#     Returns
#     -------------
#     nuclei_object:
#         mask defined extent of NU
#     """
#     median_sz = 4
#     gauss_sig = 1.34
#     dot_scale_1 = 5
#     dot_cut_1 = 0.09
#     dot_scale_2 = 2.5
#     dot_cut_2 = 0.07
#     dot_scale_3 = 1
#     dot_cut_3 = 0.01
#     filament_scale = 1
#     filament_cut = 0.15
#     min_hole_w = 0
#     max_hole_w = 25
#     small_obj_w = 3

#     return infer_lyso(
#         in_img,
#         median_sz,
#         gauss_sig,
#         dot_cut_1,
#         dot_scale_1,
#         dot_cut_2,
#         dot_scale_2,
#         dot_cut_3,
#         dot_scale_3,
#         filament_scale,
#         filament_cut,
#         min_hole_w,
#         max_hole_w,
#         small_obj_w,
#     )


# def infer_and_export_lyso(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
#     """
#     infer lyso and write inferred lyso to ome.tif file

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
#     lyso = fixed_infer_lyso(in_img)
#     out_file_n = export_inferred_organelle(lyso, "lyso", meta_dict, out_data_path)
#     print(f"inferred lyso. wrote {out_file_n}")
#     return lyso


# def lyso_spot_filter(in_img: np.ndarray) -> np.ndarray:
#     """spot filter helper function for lyso"""
#     dot_scale_1 = 5
#     dot_cut_1 = 0.09
#     dot_scale_2 = 2.5
#     dot_cut_2 = 0.07
#     dot_scale_3 = 1
#     dot_cut_3 = 0.01
#     s2_param = [[dot_scale_1, dot_cut_1], [dot_scale_2, dot_cut_2], [dot_scale_3, dot_cut_3]]
#     return dot_2d_slice_by_slice_wrapper(in_img, s2_param)


# def lyso_filiment_filter(in_img: np.ndarray) -> np.ndarray:
#     """spot filter helper function for lyso (DEPRICATED)"""
#     filament_scale = 1
#     filament_cut = 0.15
#     f2_param = [[filament_scale, filament_cut]]
#     # f2_param = [[1, 0.15]]  # [scale_1, cutoff_1]
#     return filament_2d_wrapper(in_img, f2_param)


# def get_lyso(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
#     """
#     load lyso if it exists, otherwise calculate and write to ome.tif file

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
#         start = time.time()
#         lyso = import_inferred_organelle("lyso", meta_dict, out_data_path)
#         end = time.time()
#         print(f"loaded lyso in ({(end - start):0.2f}) sec")
#     except:
#         start = time.time()
#         print("starting segmentation...")
#         lyso = infer_and_export_lyso(in_img, meta_dict, out_data_path)
#         end = time.time()
#         print(f"inferred (and exported) lyso in ({(end - start):0.2f}) sec")

#     return lyso

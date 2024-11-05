import numpy as np
from typing import Dict
from pathlib import Path
import time

from aicssegmentation.core.utils import topology_preserving_thinning

from infer_subc.core.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc.core.img import (
    fill_and_filter_linear_size,
    select_channel_from_raw,
    masked_object_thresh,
    scale_and_smooth,
    label_uint16,
    dot_filter_3
)


##########################
#  infer_golgi
##########################
def infer_golgi(
            in_img: np.ndarray,
            golgi_ch: int,
            median_sz: int,
            gauss_sig: float,
            mo_method: str,
            mo_adjust: float,
            mo_cutoff_size: int,
            min_thickness: int,
            thin_dist: int,
            dot_scale_1: float,
            dot_cut_1: float,
            dot_scale_2: float,
            dot_cut_2: float,
            dot_scale_3: float,
            dot_cut_3: float,
            dot_method: str,
            min_hole_w: int,
            max_hole_w: int,
            small_obj_w: int,
            fill_filter_method: str
        ) -> np.ndarray:

    """
    Procedure to infer golgi from linearly unmixed input.

   Parameters
    ------------
    in_img: 
        a 3d image containing all the channels
    median_sz: 
        width of median filter for signal
    mo_method: 
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
    mo_adjust: 
        Masked Object threshold `local_adjust`
    mo_cutoff_size: 
        Masked Object threshold `size_min`
    min_thinkness: 
        Half of the minimum width you want to keep from being thinned.
        For example, when the object width is smaller than 4, you don't
        want to make this part even thinner (may break the thin object
        and alter the topology), you can set this value as 2.
    thin_dist: 
        the amount to thin (has to be an positive integer). The number of
         pixels to be removed from outter boundary towards center.
    dot_scale: 
        scales (log_sigma) for dot filter (1,2, and 3)
    dot_cut: 
        threshold for dot filter thresholds (1,2,and 3)
    small_obj_w: 
        minimu object size cutoff for nuclei post-processing
    
    Returns
    -------------
    golgi_object
        mask defined extent of golgi object
    """

    ###################
    # EXTRACT
    ###################    
    golgi = select_channel_from_raw(in_img, golgi_ch)

    ###################
    # PRE_PROCESSING
    ###################    
    golgi =  scale_and_smooth(golgi,
                              median_size = median_sz, 
                              gauss_sigma = gauss_sig)
    ###################
    # CORE_PROCESSING
    ###################
    bw = masked_object_thresh(golgi, global_method=mo_method, cutoff_size=mo_cutoff_size, local_adjust=mo_adjust)

    bw_thin = topology_preserving_thinning(bw, min_thickness, thin_dist)

    bw_extra = dot_filter_3(golgi, dot_scale_1, dot_cut_1, dot_scale_2, dot_cut_2, dot_scale_3, dot_cut_3, dot_method)

    bw = np.logical_or(bw_extra, bw_thin)
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










#################################################################
########################## DEPRICATING ##########################
#################################################################


##########################
#  fixed_infer_golgi
##########################
# def fixed_infer_golgi(in_img: np.ndarray ) -> np.ndarray:
#     """
#     Procedure to infer golgi from linearly unmixed input.

#     Parameters
#     ------------
#     in_img: 
#         a 3d image containing all the channels
#     Returns
#     -------------
#    golgi_object
#         mask defined extent of golgi object
#     """

#     golgi_ch = 3
#     median_sz = 4
#     gauss_sig = 1.34
#     mo_method = 'tri'
#     mo_adjust = 1
#     mo_cutoff_size = 1200 
#     min_thickness = 1.6
#     thin_dist = 1
#     dot_scale_1 = 1.6
#     dot_cut_1 = 0.02
#     dot_scale_2 = 0
#     dot_cut_2 = 0
#     dot_scale_3 = 0
#     dot_cut_3 = 0
#     dot_method = '3D'
#     min_hole_w = 0
#     max_hole_w = 0
#     small_obj_w = 3
#     fill_filter_method = "3D"

#     return infer_golgi(
#         in_img,
#         golgi_ch,
#         median_sz,
#         gauss_sig,
#         mo_method,
#         mo_adjust,
#         mo_cutoff_size,
#         min_thickness,
#         thin_dist,
#         dot_scale_1,
#         dot_cut_1,
#         dot_scale_2,
#         dot_cut_2,
#         dot_scale_3,
#         dot_cut_3,
#         dot_method,
#         min_hole_w,
#         max_hole_w,
#         small_obj_w,
#         fill_filter_method)


# def fixed_infer_golgi(in_img: np.ndarray, cytoplasm_mask: Optional[np.ndarray] = None) -> np.ndarray:
#     """
#      Procedure to infer golgi from linearly unmixed input.

#      Parameters
#      ------------
#      in_img:
#          a 3d image containing all the channels
#      Returns
#      -------------
#     golgi_object
#          mask defined extent of golgi object
#     """

#     median_sz = 4
#     gauss_sig = 1.34
#     mo_method = "tri"
#     mo_adjust = 0.90
#     mo_cutoff_size = 1200
#     min_thickness = 1.6
#     thin = 1
#     dot_scale = 1.6
#     dot_cut = 0.02
#     small_obj_w = 3

#     return infer_golgi(
#         in_img,
#         median_sz,
#         gauss_sig,
#         mo_method,
#         mo_adjust,
#         mo_cutoff_size,
#         min_thickness,
#         thin,
#         dot_scale,
#         dot_cut,
#         small_obj_w,
#     )


# def infer_and_export_golgi(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
#     """
#     infer golgi and write inferred golgi to ome.tif file

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
#     golgi = fixed_infer_golgi(in_img)
#     out_file_n = export_inferred_organelle(golgi, "golgi", meta_dict, out_data_path)
#     print(f"inferred golgi. wrote {out_file_n}")
#     return golgi


# def get_golgi(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
#     """
#     load golgi if it exists, otherwise calculate and write to ome.tif file

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
#         golgi = import_inferred_organelle("golgi", meta_dict, out_data_path)
#     except:
#         start = time.time()
#         print("starting segmentation...")
#         golgi = infer_and_export_golgi(in_img, meta_dict, out_data_path)
#         end = time.time()
#         print(f"inferred (and exported) golgi in ({(end - start):0.2f}) sec")

#     return golgi

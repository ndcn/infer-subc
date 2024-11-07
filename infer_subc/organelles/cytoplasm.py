import numpy as np
from typing import Dict
from pathlib import Path
import time

from skimage.morphology import binary_erosion
from infer_subc.core.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc.core.img import (apply_mask, 
                                 label_bool_as_uint16, 
                                 weighted_aggregate, 
                                 scale_and_smooth, 
                                 masked_object_thresh, 
                                 fill_and_filter_linear_size,
                                 )
from infer_subc.organelles.cellmask import non_linear_cellmask_transform

### USED ###
##########################
#  infer_cytoplasm
##########################
def infer_cytoplasm(nuclei_object: np.ndarray, cellmask: np.ndarray, erode_nuclei: bool = True) -> np.ndarray:
    """
    Procedure to infer infer from linearly unmixed input. (logical cellmask AND NOT nucleus)

    Parameters
    ------------
    nuclei_object:
        a 3d image containing the nuclei object
    cellmask:
        a 3d image containing the cellmask object (mask)
    erode_nuclei:
        should we erode?

    Returns
    -------------
    cytoplasm_mask
        boolean np.ndarray

    """
    nucleus_obj = apply_mask(nuclei_object, cellmask)

    if erode_nuclei:
        cytoplasm_mask = np.logical_xor(cellmask, binary_erosion(nucleus_obj))
    else:
        cytoplasm_mask = np.logical_xor(cellmask, nucleus_obj)

    return label_bool_as_uint16(cytoplasm_mask)


def infer_and_export_cytoplasm(
    nuclei_object: np.ndarray, cellmask: np.ndarray, meta_dict: Dict, out_data_path: Path
) -> np.ndarray:
    """
    infer nucleus and write inferred nuclei to ome.tif file

    Parameters
    ------------
    nuclei_object:
        a 3d image containing the nuclei object
    cellmask:
        a 3d image containing the cellmask object (mask)
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """
    cytoplasm = infer_cytoplasm(nuclei_object, cellmask)

    out_file_n = export_inferred_organelle(cytoplasm, "cyto", meta_dict, out_data_path)
    print(f"inferred cytoplasm. wrote {out_file_n}")
    return cytoplasm


def get_cytoplasm(nuclei_obj: np.ndarray, cellmask: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    load cytoplasm if it exists, otherwise calculate and write to ome.tif file

    Parameters
    ------------
    in_img:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    cellmask:
        a 3d image containing the cellmask object (mask)
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """
    try:
        cytoplasm = import_inferred_organelle("cyto", meta_dict, out_data_path)>0
    except:
        start = time.time()
        print("starting segmentation...")
        cytoplasm = infer_and_export_cytoplasm(nuclei_obj, cellmask, meta_dict, out_data_path)
        end = time.time()
        print(f"inferred cytoplasm in ({(end - start):0.2f}) sec")

    return cytoplasm

### USED ###
##########################
# infer_cytoplasm_fromcomposite
# alternative workflow "a" 
# for cells without nuclei labels
##########################
def infer_cytoplasm_fromcomposite(in_img: np.ndarray,
                                  weights: list[int],
                                  median_sz: int,
                                  gauss_sig: float,
                                  mo_method: str,
                                  mo_adjust: float,
                                  mo_cutoff_size: int,
                                  min_hole_w: int,
                                  max_hole_w: int,
                                  small_obj_w: int,
                                  fill_filter_method: str
                                  ) -> np.ndarray:
    """
    Procedure to infer cytoplasm from linear unmixed input.

    Parameters
    ------------
    in_img: 
        a 3d image containing all the channels
    weights:
        a list of int that corresond to the weights for each channel in the composite; use 0 if a channel should not be included in the composite image
    median_sz: 
        width of median filter for cytoplasm signal
    gauss_sig: 
        sigma for gaussian smoothing of cytoplasm signal
    mo_method: 
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
    mo_adjust: 
        Masked Object threshold `local_adjust`
    mo_cutoff_size: 
        Masked Object threshold `size_min`
    max_hole_w: 
        hole filling cutoff for cytoplasm signal post-processing
    small_obj_w: 
        minimu object size cutoff for cytoplasm signal post-processing
    fill_filter_method:
        determines if hole filling and small object removal should be run 'sice-by-slice' or in '3D' 

    Returns
    -------------
    ccytoplasm_mask:
        a logical/labels object defining boundaries of cytoplasm

    """
    ###################
    # EXTRACT
    ###################
    struct_img = weighted_aggregate(in_img, *weights)

    ###################
    # PRE_PROCESSING
    ###################                         
    struct_img = scale_and_smooth(struct_img,
                                   median_size = median_sz, 
                                   gauss_sigma = gauss_sig)
    

    struct_img_non_lin = non_linear_cellmask_transform(struct_img)

    ###################
    # CORE_PROCESSING
    ###################
    struct_obj = masked_object_thresh(struct_img_non_lin, 
                                      global_method=mo_method, 
                                      cutoff_size=mo_cutoff_size, 
                                      local_adjust=mo_adjust)               

    ###################
    # POST_PROCESSING
    ###################
    struct_obj = fill_and_filter_linear_size(struct_obj, 
                                             hole_min=min_hole_w, 
                                             hole_max=max_hole_w, 
                                             min_size= small_obj_w,
                                             method=fill_filter_method)

    ###################
    # POST- POST_PROCESSING
    ###################
    cellmask_out = label_bool_as_uint16(struct_obj)

    return cellmask_out



##########################
# fixed_infer_cytoplasm_fromcomposite
# alternative workflow "a" for cells wihtout nuclei labels
##########################
def fixed_infer_cytoplasm_fromcomposite(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer cytoplasm from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img: 
        a 3d image containing all the channels
    nuclei_labels: "
        a 3d image containing the inferred nuclei

    Returns
    -------------
    cytoplasm_mask:
        a logical/labels object defining boundaries of cytoplasm
    """
    

    ###################
    # PARAMETERS
    ###################   
    weights = [0, 4, 1, 1, 2, 2]
    median_sz = 0
    gauss_sig = 0
    mo_method = "ave"
    mo_adjust = 0.2
    mo_cutoff_size = 50
    hole_min_width = 0
    hole_max_width = 30
    small_obj_w = 10
    fill_filter_method = '3D'

    cellmask_out = infer_cytoplasm_fromcomposite(in_img,
                                                weights,
                                                median_sz,
                                                gauss_sig,
                                                mo_method,
                                                mo_adjust,
                                                mo_cutoff_size,
                                                hole_min_width,
                                                hole_max_width,
                                                small_obj_w,
                                                fill_filter_method) 

    return cellmask_out

##########################
# fixed_infer_cytoplasm_fromcomposite
# alternative workflow "b" for images wihtout nuclei labels and multiple cells per view
##########################
### USED ###
def segment_cytoplasm_area(in_img: np.ndarray, 
                           global_method: str,
                           cutoff_size: int,
                           local_adjust: float,
                           min_hole_width: int,
                           max_hole_width: int,
                           small_obj_width: int,
                           fill_filter_method: str):
    """ 
    Function for segmenting the cytoplasmic area from a fluorescent image

    Parameters:
    ----------
    in_img: np.ndarray, 
        fluorescence image (single channel, ZYX array) of the cytoplasm to get segmented
    global_method: str,
        masked object threshold method; options: 'med', 'tri', 'ave'
    cutoff_size: int,
        object cutoff size for the MO threshold method
    local_adjust: float,
        adjustment value for the MO threshold method
    min_hole_width: int,
        smallest sized hole to fill in the final mask
    max_hole_width: int,
        largest sized hole to fill in the final mask
    small_obj_width: int,
        size of the smallest object to be included in the mask; small objects are removed
    fill_filter_method: str
        fill holes and remove small objects in '3D' or 'slice_by_slice'


    """
    # create cytoplasm mask
    bw_cyto = masked_object_thresh(in_img, 
                            global_method=global_method, 
                            cutoff_size=cutoff_size, 
                            local_adjust=local_adjust)
    
    # fill holes and filter small objects from the raw mask
    cleaned_cyto = fill_and_filter_linear_size(bw_cyto, 
                                            hole_min=min_hole_width, 
                                            hole_max=max_hole_width, 
                                            min_size= small_obj_width,
                                            method=fill_filter_method)
    
    # create a boolean mask
    cyto_semantic_seg = cleaned_cyto.astype(bool)

    return cyto_semantic_seg
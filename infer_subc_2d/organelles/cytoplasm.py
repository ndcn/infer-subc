import numpy as np
from typing import Dict
from pathlib import Path
import time

from skimage.morphology import binary_erosion
from infer_subc_2d.core.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc_2d.core.img import *


##########################
#  infer_cytoplasm
##########################
def infer_cytoplasm(nuclei_object: np.ndarray, soma_mask: np.ndarray, erode_nuclei: bool = True) -> np.ndarray:
    """
    Procedure to infer infer from linearly unmixed input. (logical cellmask AND NOT nucleus)

    Parameters
    ------------
    nuclei_object:
        a 3d image containing the nuclei object
    soma_mask:
        a 3d image containing the cellmask object (mask)
    erode_nuclei:
        should we erode?

    Returns
    -------------
    cytoplasm_mask
        boolean np.ndarray

    """
    nucleus_obj = apply_mask(nuclei_object, soma_mask)

    if erode_nuclei:
        cytoplasm_mask = np.logical_xor(soma_mask, binary_erosion(nucleus_obj))
    else:
        cytoplasm_mask = np.logical_xor(soma_mask, nucleus_obj)

    return cytoplasm_mask


def infer_and_export_cytoplasm(
    nuclei_object: np.ndarray, soma_mask: np.ndarray, meta_dict: Dict, out_data_path: Path
) -> np.ndarray:
    """
    infer nucleus and write inferred nuclei to ome.tif file

    Parameters
    ------------
    nuclei_object:
        a 3d image containing the nuclei object
    soma_mask:
        a 3d image containing the cellmask object (mask)
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """
    cytoplasm = infer_cytoplasm(nuclei_object, soma_mask)

    out_file_n = export_inferred_organelle(cytoplasm, "cytoplasm", meta_dict, out_data_path)
    print(f"inferred cytoplasm. wrote {out_file_n}")
    return cytoplasm


def get_cytoplasm(nuclei_obj: np.ndarray, soma_mask: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    load cytoplasm if it exists, otherwise calculate and write to ome.tif file

    Parameters
    ------------
    in_img:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    soma_mask:
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
        cytoplasm = import_inferred_organelle("cytoplasm", meta_dict, out_data_path)
    except:
        start = time.time()
        print("starting segmentation...")
        cytoplasm = infer_and_export_cytoplasm(nuclei_obj, soma_mask, meta_dict, out_data_path)
        end = time.time()
        print(f"inferred cytoplasm in ({(end - start):0.2f}) sec")

    return cytoplasm

##########################
#  infer_cytoplasm_from_composite
##########################
def infer_cytoplasm_from_composite(in_img: np.ndarray,
                                    weights: list,
                                    low_level_min_size: int,
                                    thresh_method: str,
                                    local_adjust: float,
                                    holefill_min: int,
                                    holefill_max: int,
                                    small_object_width: int,
                                    method: str,
                                    connectivity: int
                                                            ) -> np.ndarray:
    """
    Procedure to infer 3D cytoplasm (cell area without the nucleus) segmentation from multichannel z-stack input.
    This can be used when segmenting the cytoplasm from a stain that fills the cytoplasm, but not the nuclear area.

    Parameters
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels
    weights: list
        a list of weights for each channel in the image to create a merged image; 
        set to 0 if the channel should not be used
    low_level_min_size: int
        global threshold size restriction
    thresh_method: str
        global threshold method, 'ave', 'tri', or 'med'
    local_adjust: float
        local threshold factor
    holefill_min: int
        minimum hole size to fill
    holefill_max: int
        maximum hole size to fill
    small_object_width: int
        diameter of largest object to remove; all smaller object will be removed
    method: str
        '3D' or 'slice-by-slice'
    connectivity: int
        connectivity to determine indepedent object for size filtering

    Returns
    -------------
    cytoplasm_mask
        mask defined extent of cytoplasm (cell without the nuclear area)
    
    """

    ###################
    # INPUT
    ###################
    struct_img_raw = make_aggregate(in_img, weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], scale_min_max=False)

    ###################
    # PRE_PROCESSING
    ###################           
    composite = log_rescale_wrapper(struct_img_raw)


    ###################
    # CORE_PROCESSING
    ###################
    bw = masked_object_thresh(composite, th_method=thresh_method, cutoff_size=low_level_min_size, th_adjust=local_adjust)

    ###################
    # POST_PROCESSING
    ###################
    cleaned_img = fill_and_filter_linear_size(bw, hole_min=holefill_min, hole_max=holefill_max, min_size=small_object_width, method=method, connectivity=connectivity)

    ###################
    # RENAMING
    ###################
    cytoplasm_mask = cleaned_img.astype(dtype=int)

    return cytoplasm_mask

##########################
#  fixed_infer_cytoplasm_from_composite
##########################
def fixed_infer_cytoplasm_from_composite(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer soma from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels
    soma_mask: np.ndarray
        mask
 
    Returns
    -------------
    nuclei_object
        mask defined extent of NU
    
    """
    weights = [0,4,1,1,2,2]
    low_level_min_size = 50
    thresh_method = 'ave'
    local_adjust = 0.05
    holefill_min = 0
    holefill_max = 30
    small_object_width = 10
    method = '3D'
    connectivity = 1
    

    return infer_cytoplasm_from_composite(in_img,
                                            weights,
                                            low_level_min_size,
                                            thresh_method,
                                            local_adjust,
                                            holefill_min,
                                            holefill_max,
                                            small_object_width,
                                            method,
                                            connectivity)

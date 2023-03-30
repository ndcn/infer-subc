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


def infer_cytoplasm_fromaggr(img_in: np.ndarray,
                             th_method: str,
                             th_cutoff_sz: int,
                             th_adjust: int,
                             smallhole_min_sz: int,
                             smallhole_max_sz: int,
                             smallobj_max_sz: int,
                             slice_or_3D: str,
                             filter_connectivity: int) -> np.ndarray:
    """ 
    Procedure to mask the cytoplasmic area of a cell (based on 'MO' tresholding from 'aicssegmentation'). Includes hole and size filtering.

    Parameters
    ----------
    img_in: np.dnarray
        Grey scale fluorescence image of cytoplasmic area. Image should already be preprocessed (smoothing, rescaling, etc.)
    th_method: str
        Threshold method for masked_object_threshold() function in infer_subc_2D.core.img.
        Options: "triangle", "median", and "ave_tri_med"
    th_cutoff_sz: str 
        Maximum object size for masked_object_threshold() function in infer_subc_2D.core.img.
    th_adjust: str
        Local threshold adjustment value for masked_object_threshold() function in infer_subc_2D.core.img.
    smallhole_min_sz: int
        Minimum size of hole to fill in cytoplasm mask.
    smallhole_max_sz: int
        Maximum size of hole to fill in cytoplasm mask.
    smallobj_max_sz: int
        Maximum small object size to remove from cytoplasm mask.
    slices_or_3D: str
        Should the filtering be performed slice-by-slice or in 3D?
        Options: 'slice-by-slice', or '3D'
    filter_connectivity: int
        Connectivity of objects in cytoplasm mask for filtering function

    Output
    ------
    cytoplasm_mask: np.ndarray
        binary image of cyplasmic area (cell area without nucleus)
    """
    ###################
    # CORE_PROCESSING
    ###################
    cyto_bw = masked_object_thresh(img_in, th_method=th_method, cutoff_size=th_cutoff_sz, th_adjust=th_adjust)
    
    ###################
    # POST_PROCESSING
    ###################
    cyto_cleaned_img = fill_and_filter_linear_size(cyto_bw, hole_min=smallhole_min_sz, hole_max=smallhole_max_sz, min_size=smallobj_max_sz, method=slice_or_3D, connectivity=filter_connectivity)
    cytoplasm_mask = cyto_cleaned_img

    return cytoplasm_mask 



##########################
#  infer_cytoplasm_fromaggr_wrapper
##########################
def infer_cytoplasm_fromaggr_wrapper(in_img: np.ndarray,
                                    weights: list,
                                    th_method: str,
                                    th_cutoff_sz: int,
                                    th_adjust: int,
                                    smallhole_min_sz: int,
                                    smallhole_max_sz: int,
                                    smallobj_max_sz: int,
                                    slice_or_3D: str,
                                    filter_connectivity: int) -> np.ndarray:
    """
    Wrapper function for infer_cytoplasm_fromaggr(). 
    Includes preprocessing steps: create weighted aggregate image, rescale, log, rescale

    Parameters
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels
    weights: list
        a list of weights for each channel in the image to create a merged image; 
        set to 0 if the channel should not be used
    img_in: np.dnarray
        Grey scale fluorescence image of cytoplasmic area. Image should already be preprocessed (smoothing, rescaling, etc.)
    th_method: str
        Threshold method for masked_object_threshold() function in infer_subc_2D.core.img.
        Options: "triangle", "median", and "ave_tri_med"
    th_cutoff_sz: str 
        Maximum object size for masked_object_threshold() function in infer_subc_2D.core.img.
    th_adjust: str
        Local threshold adjustment value for masked_object_threshold() function in infer_subc_2D.core.img.
    smallhole_min_sz: int
        Minimum size of hole to fill in cytoplasm mask.
    smallhole_max_sz: int
        Maximum size of hole to fill in cytoplasm mask.
    smallobj_max_sz: int
        Maximum small object size to remove from cytoplasm mask.
    slices_or_3D: str
        Should the filtering be performed slice-by-slice or in 3D?
        Options: 'slice-by-slice', or '3D'
    filter_connectivity: int
        Connectivity of objects in cytoplasm mask for filtering function

    Output
    ------
    cytoplasm_mask: np.ndarray
        binary image of cytoplasmic area  (cell area without nucleus)
    
    """
    ###################
    # INPUT
    ###################
    composite = make_aggregate(in_img, weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], scale_min_max=False)

    ###################
    # PRE_PROCESSING
    ###################           
    struct_img = log_rescale_wrapper(composite)

    ###################
    # CORE_PROCESSING
    ###################
    cleaned_img = infer_cytoplasm_fromaggr(struct_img,
                             th_method=th_method,
                             th_cutoff_sz=th_cutoff_sz,
                             th_adjust=th_adjust,
                             smallhole_min_sz= smallhole_min_sz,
                             smallhole_max_sz=smallhole_max_sz,
                             smallobj_max_sz=smallobj_max_sz,
                             slice_or_3D=slice_or_3D,
                             filter_connectivity=filter_connectivity)
    
    ###################
    # RENAMING
    ###################
    cytoplasm_mask = cleaned_img.astype(dtype=int)

    return cytoplasm_mask


##########################
#  fixed_infer_cytoplasm_from_composite
##########################
def fixed_infer_cytoplasm_fromaggr_wrapper(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer soma from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ----------
    in_img: np.ndarray
        Raw image for processing using the infer_cytoplasm_fromaggr_wrapper() function.
        Parameters are set for primary neuron images that were deconvolved.
 
    Returns
    -------------
    cytoplasm_mask
    
    """
    weights = [0,4,1,1,2,2]
    th_method= 'ave'
    th_cutoff_sz= 50
    th_adjust= 0.05
    smallhole_min_sz= 0
    smallhole_max_sz= 30
    smallobj_max_sz= 10
    slice_or_3D= '3D'
    filter_connectivity= 1


    return infer_cytoplasm_fromaggr_wrapper(in_img,
                                            weights,
                                            th_method,
                                            th_cutoff_sz,
                                            th_adjust,
                                            smallhole_min_sz,
                                            smallhole_max_sz,
                                            smallobj_max_sz,
                                            slice_or_3D,
                                            filter_connectivity)
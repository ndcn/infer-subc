import numpy as np
from typing import Union, Dict
from pathlib import Path
import time
import skimage

from infer_subc_2d.core.file_io import (
    export_inferred_organelle,
    import_inferred_organelle,
    import_inferred_organelle_AICS,
)
from infer_subc_2d.core.img import (
    fill_and_filter_linear_size,
    apply_log_li_threshold,
    select_channel_from_raw,
    scale_and_smooth,
    apply_mask,
    hole_filling_linear_size
)
from infer_subc_2d.constants import NUC_CH


##########################
#  infer_nuclei_fromlabel
##########################
def infer_nuclei_fromlabel(
    in_img: np.ndarray,
    nuc_ch: Union[int, None],
    median_sz: int,
    gauss_sig: float,
    thresh_factor: float,
    thresh_min: float,
    thresh_max: float,
    max_hole_w: int,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer nuclei from linearly unmixed input.

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels; np.ndarray
    median_sz:
        width of median filter for signal
    gauss_sig:
        sigma for gaussian smoothing of  signal
    thresh_factor:
        adjustment factor for log Li threholding
    thresh_min:
        abs min threhold for log Li threholding
    thresh_max:
        abs max threhold for log Li threholding
    max_hole_w:
        hole filling cutoff for nuclei post-processing0
    small_obj_w:
        minimu object size cutoff for nuclei post-processing

    Returns
    -------------
    nuclei_object
        mask defined extent of NU

    """

    ###################
    # PRE_PROCESSING
    ###################
    if nuc_ch is None:
        nuc_ch = NUC_CH
    print(f" in_img = {in_img.shape}")

    nuclei = select_channel_from_raw(in_img, nuc_ch)

    print(f"nuclei size = {nuclei.shape}")

    nuclei = scale_and_smooth(nuclei, median_sz=median_sz, gauss_sig=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    nuclei_object = apply_log_li_threshold(
        nuclei, thresh_factor=thresh_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )

    ###################
    # POST_PROCESSING
    ###################
    nuclei_object = fill_and_filter_linear_size(nuclei_object, hole_min=0, hole_max=max_hole_w, min_size=small_obj_w)
    return nuclei_object


##########################
#  fixed_infer_nuclei
##########################
def fixed_infer_nuclei(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer cellmask from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    Returns
    -------------
    nuclei_object
        inferred nuclei

    """
    nuc_ch = NUC_CH
    median_sz = 4
    gauss_sig = 1.34
    thresh_factor = 0.9
    thresh_min = 0.1
    thresh_max = 1.0
    max_hole_w = 25
    small_obj_w = 15

    return infer_nuclei_fromlabel(
        in_img, nuc_ch, median_sz, gauss_sig, thresh_factor, thresh_min, thresh_max, max_hole_w, small_obj_w
    )


def infer_and_export_nuclei(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    infer nuclei and write inferred nuclei to ome.tif file

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
    nuclei = fixed_infer_nuclei(in_img)

    out_file_n = export_inferred_organelle(nuclei, "nuclei", meta_dict, out_data_path)
    print(f"inferred nuclei. wrote {out_file_n}")
    return nuclei


def get_nuclei(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    load nucleus if it exists, otherwise calculate and write to ome.tif file

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
        nuclei = import_inferred_organelle("nuclei", meta_dict, out_data_path)
    except:
        start = time.time()
        print("starting segmentation...")
        nuclei = infer_and_export_nuclei(in_img, meta_dict, out_data_path)
        end = time.time()
        print(f"inferred nuclei in ({(end - start):0.2f}) sec")

    return nuclei


##########################
#  infer_nucleus_cellmask_from_cytoplasm()
##########################
def infer_nucleus_cellmask_from_cytoplasm(cytoplasm_mask: np.ndarray,
                                            nucleus_min_sz: int,
                                            nucleus_max_sz: int,
                                            nucleus_fill_2D: bool,
                                            nucmask_holefill_min: int,
                                            nucmask_holefill_max: int,
                                            nucmask_small_object_width: int,
                                            nucmask_slice_or_3D: str,
                                            nucmask_connectivity: int
                                                            ) -> np.ndarray:
    """
    Procedure to infer the nucleus and cell masks from the cytoplasm mask (cell area without nuclei). 
    This could be used in the case where the nuclei are not stained and other organelle markers fill the cytoplasm, but not the nuclear space.

    Parameters
    ------------
    cytoplasm_mask: np.ndarray
        3D (XYZ) binary mask of the cytoplasm (whole cell area without the nucleus)
    nucleus_min_sz: int
        minimum size of the nucleus that will be filled to created the cell mask
    nucleus_max_sz: int
        maximum size of the nucleus that will be filled to created the cell mask
    nucleus_fill_2D: bool
        Should the hole filling occur slice-by-slice or in 3D?
        Options: 'slice-by-slice' or '3D'
    nucmask_holefill_min: int
        minimum size of small holes to fill to clean up the nucleus mask        
    nucmask_holefill_max: int
        maximum size of small holes to fill to clean up the nucleus mask         
    nucmask_small_object_width: int
        size of small objects to remove to clean up the nucleus mask
    nucmask_slice_or_3D: str
        Should the hole filling and size filtering occur slice-by-slice or in 3D?
        Options: 'slice-by-slice' or '3D'
    nucmask_connectivity: int
        fill and filter connectivity between objects; should be an integer between 1 and ndim of image

    Returns
    -------------
    cell_mask
        mask defining extent of the entire cell
    nucleus_mask
        mask defining extent of the nucleus
    """

    ###################
    # PRE_PROCESSING
    ###################                
    cytoplasm_dilated = skimage.morphology.binary_dilation(cytoplasm_mask)

    ###################
    # CORE_PROCESSING
    ###################
    # Cell mask
    cytoplasm_filled = hole_filling_linear_size(cytoplasm_dilated,  hole_min=nucleus_min_sz, hole_max=nucleus_max_sz, fill_2d=nucleus_fill_2D)
    cytoplasm_eroded = skimage.morphology.binary_erosion(cytoplasm_filled)
    cell_bw = cytoplasm_eroded

    # Nucleus
    nuclei_xor = np.logical_xor(cytoplasm_mask, cell_bw)

    ###################
    # POST_PROCESSING
    ###################
    # Nucleus
    nuc_cleaned_img = fill_and_filter_linear_size(nuclei_xor, hole_min=nucmask_holefill_min, hole_max=nucmask_holefill_max, min_size=nucmask_small_object_width, method=nucmask_slice_or_3D, connectivity=nucmask_connectivity)
    
    # Cell mask
    extra_from_dilate = np.logical_xor(nuc_cleaned_img, nuclei_xor)
    cell_mask_cleaned_img = np.logical_xor(cell_bw, extra_from_dilate)

    ###################
    # RENAMING
    ###################
    cell_mask = cell_mask_cleaned_img.astype(dtype=int)
    nucleus_mask = nuc_cleaned_img.astype(dtype=int)

    masks = np.stack((nucleus_mask, cytoplasm_mask, cell_mask), axis=0)

    return masks

    # return cell_mask, nucleus_mask


##########################
#  fixed_infer_nucleus_cellmask_from_cytoplasm
##########################
def fixed_infer_nucleus_cellmask_from_cytoplasm(cytoplasm_mask: np.ndarray) -> np.ndarray:
    """
    Procedure to infer cellmask from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels
 
    Returns
    -------------
    cell_mask
        mask defining extent of the entire cell
    nucleus_mask
        mask defining extent of the nucleus
    
    """
    nucleus_min_sz = 0
    nucleus_max_sz = 500
    nucleus_fill_2D = False

    nucmask_holefill_min = 0
    nucmask_holefill_max = 0
    nucmask_small_object_width = 20
    nucmask_slice_or_3D = '3D'
    nucmask_connectivity = 3

    return infer_nucleus_cellmask_from_cytoplasm(cytoplasm_mask,
                                                nucleus_min_sz,
                                                nucleus_max_sz,
                                                nucleus_fill_2D,
                                                nucmask_holefill_min,
                                                nucmask_holefill_max,
                                                nucmask_small_object_width,
                                                nucmask_slice_or_3D,
                                                nucmask_connectivity)


# def infer_nuclei_fromlabel_AICS(in_img: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
#     """
#     load nucleus if it exists, otherwise calculate and write to ome.tif file

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
#         nuclei = import_inferred_organelle_AICS("nuclei", meta_dict, out_data_path)
#     except:
#         start = time.time()
#         print("starting nuclei segmentation...")
#         nuclei = fixed_infer_nuclei(in_img)
#         out_file_n = export_inferred_organelle_AICS(nuclei, "nuclei", meta_dict, out_data_path)
#         end = time.time()
#         print(f"inferred and saved nuclei AICS in ({(end - start):0.2f}) sec")

#     return nuclei

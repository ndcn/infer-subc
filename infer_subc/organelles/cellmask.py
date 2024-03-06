from typing import Dict, Union
from pathlib import Path
import time
import numpy as np
from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.utils import hole_filling

from skimage.filters import scharr
from skimage.measure import label
from skimage.morphology import disk, closing, binary_dilation, dilation, binary_erosion

from infer_subc.core.img import label_bool_as_uint16
from infer_subc.constants import (
    TEST_IMG_N,
    NUC_CH,
    LYSO_CH,
    MITO_CH,
    GOLGI_CH,
    PEROX_CH,
    ER_CH,
    LD_CH,
    PM_CH,
    RESIDUAL_CH,
)
from infer_subc.core.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc.core.img import (
    masked_object_thresh,
    log_transform,
    min_max_intensity_normalization,
    median_filter_slice_by_slice,
    scale_and_smooth,
    weighted_aggregate,
    masked_inverted_watershed,
    fill_and_filter_linear_size,
    get_max_label,
    get_interior_labels,
    select_channel_from_raw,
    threshold_otsu_log,
    inverse_log_transform,
)


def raw_cellmask_fromaggr(img_in: np.ndarray, scale_min_max: bool = True) -> np.ndarray:
    """define cellmask image
    CELLMASK_W = (6.,1.,2.)
    CELLMASK_CH = (LYSO_CH,ER_CH,GOLGI_CH)

    Parameters
    ------------
    img_in
        a 3d image
    scale_min_max:
        scale to [0,1] if True. default True

    Returns
    -------------
        np.ndarray scaled aggregate

    """
    weights = (0, 6, 0, 2, 0, 1)
    if scale_min_max:
        return min_max_intensity_normalization(weighted_aggregate(img_in, *weights))
    else:
        return weighted_aggregate(img_in, *weights)


def non_linear_cellmask_transform(in_img):
    """non-linear distortion to fill out cellmask
    log + edge of smoothed composite

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    Returns
    -------------
        np.ndarray scaled aggregate
    """
    # non-Linear processing
    log_img, d = log_transform(in_img.copy())
    log_img = min_max_intensity_normalization(log_img)
    return min_max_intensity_normalization(scharr(log_img)) + log_img


def choose_max_label_cellmask_union_nucleus(cellmask_img: np.ndarray, 
                                            cellmask_obj: np.ndarray, 
                                            nuclei_labels: np.ndarray, 
                                            watershed_method: str = 'slice-by-slice',
                                            interior_labels_only: bool = True
                                            ) -> np.ndarray:
    """get cellmask UNION nuclei for largest signal label

        Parameters
    ------------
    cellmask_img:
        the cellmask image intensities
    cellmask_obj:
        thresholded cellmask mask
    nuclei_labels:
        inferred nuclei labels (np.uint16)
    watershed_method:
        determines if the watershed should be run 'sice-by-slice' or in '3D' 

    Returns
    -------------
        boolean np.ndarray of cellmask+nuc corresponding to the label of largest total cellmask signal

    """

    cellmask_labels = masked_inverted_watershed(cellmask_img, nuclei_labels, cellmask_obj, method=watershed_method)

    # should we restrict to interior nuclear labels?
    # get_interior_labels(nuclei_object)
    # would need to update get_max_label to only choose the labels in get_interior_label
    target_labels = get_interior_labels(nuclei_labels) if interior_labels_only else None

    keep_label = get_max_label(cellmask_img, cellmask_labels, target_labels=target_labels)

    cellmask_out = np.zeros_like(cellmask_labels)
    cellmask_out[cellmask_labels == keep_label] = 1
    cellmask_out[nuclei_labels == keep_label] = 1

    return cellmask_out > 0

def rescale_intensity(in_img: np.ndarray):
    #rescales the intensity of input image on a scale of 0 to 10
    out_img = ((in_img - in_img.min())/(in_img.max() - in_img.min()))*10
    return out_img

def create_composite(in_img: np.ndarray,
                     weights: list[float],
                     invert_PM: bool=False):
    out_img = np.zeros_like(in_img[0]).astype(np.double)
    for channel, weight in enumerate(weights):
        if weight > 0:
            if (channel == PM_CH)&(invert_PM):
                out_img += weight * abs(np.max(in_img[PM_CH]) - in_img[PM_CH])
            else:
                out_img += weight * rescale_intensity(in_img[channel])
    return out_img

##########################
# infer_mask_from_membrane
##########################
def infer_cellmask_from_membrane(in_img: np.ndarray,
                                weights: list[float],
                                nuclei_labels: np.ndarray,
                                invert_PM_A: bool=False,
                                invert_PM_B: bool=False,
                                close_fp_size: int=15,
                                thresh_method: str="med",
                                cutoff_A: int=0,
                                cutoff_B: int=0,
                                thresh_adj_A: float=1.0,
                                thresh_adj_B: float=1.0,
                                nuc_fp_size: int=0,
                                hole_min: int=0, 
                                hole_max: int=0,
                                small_obj: int=0,
                                bound_pm_A: bool=False,
                                bound_pm_B: bool=False,
                                adj_pm: float=1.0,
                                watershed_method: str="3D"
                                ):
    """
    Procedure to infer cellmask fom a composite including plasma membrane

    Parameters
    ------------
    in_img:
        a 3D image containing all channels (Channel, Z, Y, X)
    weights:
        a list of floats involved in the weight used for each channel; zero is used if channel is to be excluded
    invert_PM_A:
        a true/false statement determining whether or not to invert PM when creating composite A
    invert_PM_B:
        a true/false statement determining whether or not to invert PM when creating composite B
    close_fp_size:
        an integer value that determines the size of the disk used in closing the composite image
    thresh_method:
        which method to use for calculating global threshold. Options include:
        "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
        "ave" refers the average of "triangle" threshold and "mean" threshold.
    cutoff_A:
        thresholding composite A's minimum size
    cutoff_B:
        thresholding composite B's minimum size
    thresh_adj_A:
        a decimal number used for adjusting the threshold of composite A
    trhesh_adj_B:
        a decimal number used for adjusting the threshold of composite B
    hole_min_thresh:
        minimum size for filling holes during thresholding
    hole_max_thresh:
        maximum size for filling holes during thresholding
    small_obj:
        minimum object size cutoff for thresholding
    bound_pm_A:
        a true/false statement determining whether or not to enforce the boundaries of the plasma membrane in watershed A
    bound_pm_B:
        a true/false statement determining whether or not to enforce the boundaries of the plasma membrane in watershed B
    adj_pm:
        a decimal number used for adjusting the threshold of the plasma membrane
    watershed_method:
        determines if the watershed should be run 'sice-by-slice' or in '3D'
    """
    #################
    # EXTRACT IMAGES
    #################
    composite_A = create_composite(in_img, weights, invert_PM_A)
    composite_B = create_composite(in_img, weights, invert_PM_B)

    #################
    # PRE-PROCESSING
    #################
    closed_A = np.zeros_like(composite_A)
    closed_B = np.zeros_like(composite_B)
    close_fp = disk(close_fp_size)
    for z in range(len(composite_A)):
        closed_A[z] = closing(composite_A[z].copy(), footprint=close_fp)
        closed_B[z] = closing(composite_B[z].copy(), footprint=close_fp)
    
    #################
    # CORE-PROCESSING
    #################
        #unsure if min_max_intensity_normalization() is required here, possible spot to increase speed
    closed_threshed_A = masked_object_thresh(min_max_intensity_normalization(closed_A.copy()),
                                             global_method=thresh_method,
                                             cutoff_size=cutoff_A,
                                             local_adjust=thresh_adj_A).astype(bool)
    closed_threshed_B = masked_object_thresh(min_max_intensity_normalization(closed_B.copy()),
                                             global_method=thresh_method,
                                             cutoff_size=cutoff_B,
                                             local_adjust=thresh_adj_B).astype(bool)
        #determining the nuclei for watershedding
    keep_nuc = get_max_label((closed_threshed_A), dilation(nuclei_labels))
    single_nuc = np.zeros_like(nuclei_labels)
    single_nuc[nuclei_labels == keep_nuc] = 1

        #adding nuclei to the threshold
    if nuc_fp_size > 0:
        nuc_fp = disk(nuc_fp_size)
        for z in range(len(closed_threshed_A)):
            closed_threshed_A[z] += binary_dilation(single_nuc.astype(bool)[z], footprint=nuc_fp)
            closed_threshed_B[z] += binary_dilation(single_nuc.astype(bool)[z], footprint=nuc_fp)
    else:
        for z in range(len(closed_threshed_A)):
            closed_threshed_A[z] += binary_dilation(single_nuc.astype(bool)[z])
            closed_threshed_B[z] += binary_dilation(single_nuc.astype(bool)[z])
    

    #################
    # POST-PROCESSING
    #################
    filled_A = fill_and_filter_linear_size(closed_threshed_A,
                                           hole_min=hole_min,
                                           hole_max=hole_max,
                                           min_size=small_obj)
    filled_B = fill_and_filter_linear_size(closed_threshed_B,
                                           hole_min=hole_min,
                                           hole_max=hole_max,
                                           min_size=small_obj)
    if bound_pm_A:
        pm_img = select_channel_from_raw(in_img, PM_CH)
        pm_image, d = log_transform(pm_img.copy())
        pm_thresh = threshold_otsu_log(pm_image) 
        invert_pm_obj = np.invert(pm_img > (inverse_log_transform(pm_thresh, d) * adj_pm))
        mask_A = np.zeros_like(invert_pm_obj)
        mask_A[(invert_pm_obj == filled_A) & 
                 (invert_pm_obj == 1) & 
                 (filled_A == 1)] = 1
        mask_A = closing(mask_A)
    else:
        mask_A = filled_A.copy()
    if bound_pm_B:
        pm_img = select_channel_from_raw(in_img, PM_CH)
        pm_image, d = log_transform(pm_img.copy())
        pm_thresh = threshold_otsu_log(pm_image) 
        invert_pm_obj = np.invert(pm_img > (inverse_log_transform(pm_thresh, d) * adj_pm))
        mask_B = np.zeros_like(invert_pm_obj)
        mask_B[(invert_pm_obj == filled_B) & 
                 (invert_pm_obj == 1) & 
                 (filled_B == 1)] = 1
        mask_B = closing(mask_B)
    else:
        mask_B = filled_B.copy()

    ######################
    # POST-POST PROCESSING
    ######################
    cm_A = masked_inverted_watershed(closed_threshed_A, 
                                     single_nuc, 
                                     mask_A,
                                     method=watershed_method)
    cm_B = masked_inverted_watershed(closed_threshed_B, 
                                     single_nuc, 
                                     mask_B,
                                     method=watershed_method)
    cm_combo = cm_A.astype(bool) + cm_B.astype(bool)
    for z in range(len(cm_combo)):
        cm_combo[z] = binary_dilation(cm_combo.copy()[z], footprint=close_fp)
    cm_combo = hole_filling(cm_combo.copy(), hole_min=hole_min, hole_max=hole_max, fill_2d=True)
    for z in range(len(cm_combo)):
        cm_combo[z] = binary_erosion(cm_combo.copy()[z], footprint=close_fp)
    return cm_combo

##########################
# 1. infer_cellmask
##########################
def infer_cellmask_fromcomposite(in_img: np.ndarray,
                                 weights: list[int],
                                 nuclei_labels: np.ndarray,
                                 median_sz: int,
                                 gauss_sig: float,
                                 mo_method: str,
                                 mo_adjust: float,
                                 mo_cutoff_size: int,
                                 min_hole_w: int,
                                 max_hole_w: int,
                                 small_obj_w: int,
                                 fill_filter_method: str,
                                 watershed_method: str
                                 ) -> np.ndarray:
    """
    Procedure to infer cellmask from linear unmixed input.

    Parameters
    ------------
    in_img: 
        a 3d image containing all the channels (CZYX)
    weights:
        a list of int that corresond to the weights for each channel in the composite; use 0 if a channel should not be included in the composite image
    nuclei_labels: 
        a 3d image containing the inferred nuclei labels
    median_sz: 
        width of median filter for _cellmask_ signal
    gauss_sig: 
        sigma for gaussian smoothing of _cellmask_ signal
    mo_method: 
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
    mo_adjust: 
        Masked Object threshold `local_adjust`
    mo_cutoff_size: 
        Masked Object threshold `size_min`
    min_hole_w: 
        minimum size for hole filling for cellmask signal post-processing
    max_hole_w: 
        hole filling cutoff for cellmask signal post-processing
    small_obj_w: 
        minimum object size cutoff for cellmask signal post-processing
    fill_filter_method:
        determines if small hole filling and small object removal should be run 'sice-by-slice' or in '3D'
    watershed_method:
        determines if the watershed should be run 'sice-by-slice' or in '3D' 

    Returns
    -------------
    cellmask_mask:
        a logical/labels object defining boundaries of cellmask

    """
    ###################
    # EXTRACT
    ###################
    struct_img = weighted_aggregate(in_img, *weights)

    ###################
    # PRE_PROCESSING
    ###################                         
    struct_img =  scale_and_smooth(struct_img,
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
    cellmask_out = choose_max_label_cellmask_union_nucleus(struct_img, 
                                                           struct_obj, 
                                                           nuclei_labels, 
                                                           watershed_method=watershed_method) 

    return label_bool_as_uint16(cellmask_out)


def fixed_infer_cellmask_fromcomposite(in_img: np.ndarray, nuclei_labels: np.ndarray) -> np.ndarray:
    """
    Procedure to infer cellmask from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img: 
        a 3d image containing all the channels
    nuclei_labels: 
        a 3d image containing the inferred nuclei

    Returns
    -------------
    cellmask_mask:
        a logical/labels object defining boundaries of cellmask
    """
    

    ###################
    # PARAMETERS
    ###################   
    weights = [0,0,0,3,3,2]
    median_sz = 10
    gauss_sig = 1.34
    mo_method = "med"
    mo_adjust = 0.3
    mo_cutoff_size = 150
    hole_min_width = 0
    hole_max_width = 50
    small_obj_w = 45
    fill_filter_method = '3D'
    watershed_method = '3D'

    cellmask_out = infer_cellmask_fromcomposite(in_img,
                                                weights,
                                                nuclei_labels,
                                                median_sz,
                                                gauss_sig,
                                                mo_method,
                                                mo_adjust,
                                                mo_cutoff_size,
                                                hole_min_width,
                                                hole_max_width,
                                                small_obj_w,
                                                fill_filter_method,
                                                watershed_method) 
    
    return cellmask_out.astype(np.uint8)


##########################
#  infer_cellmask_fromcytoplasm
##########################
def infer_cellmask_fromcytoplasm(cytoplasm_mask: np.ndarray,
                                  nucleus_mask: np.ndarray,
                                  min_hole_width: int,
                                  max_hole_width: int,
                                  small_obj_width: int,
                                  fill_filter_method: str
                                  ) -> np.ndarray:
    """
    Procedure to infer 3D nuclei segmentation from multichannel z-stack input.

    Parameters
    ------------
    cytoplasm_mask: np.ndarray
        3D image containing the mask of the cytoplasm
    nucleus_mask: np.ndarray
        3D image containing the mask of the nucleus
    min_hole_width: int
        minimum size of holes to fill in final cell mask
    max_hole_width: int,
        maximum size of holes to fill in final cell mask
    small_obj_w: int
        minimum object size cutoff to remove from final cell mask; likely not required since small objects were removed from cytoplasm mask
    fill_method: str
        method for fill and filter; either "3D" or "slice_by_slice"

    Returns
    -------------
    cell_mask
        mask defined extent of the entire cell
    
    """

    ###################
    # CORE_PROCESSING
    ###################
    cell = np.logical_or(nucleus_mask, cytoplasm_mask)

    ###################
    # POST_PROCESSING
    ###################
    cleaned_img = fill_and_filter_linear_size(cell, 
                                              hole_min=min_hole_width, 
                                              hole_max=max_hole_width, 
                                              min_size=small_obj_width, 
                                              method=fill_filter_method)

    ###################
    # LABELING
    ###################
    cell_mask = label_bool_as_uint16(cleaned_img)

    return cell_mask


##########################
#  fixed_infer_cellmask_fromcytoplasm
##########################
def fixed_infer_cellmask_fromcytoplasm(cytoplasm_mask: np.ndarray,
                                        nucleus_mask:np.ndarray) -> np.ndarray:
    """
    Procedure to infer cellmask from the cytoplasm mask

    Parameters
    ------------
    in_img: np.ndarray
        a 3d image containing cytoplasm segmentation
 
    Returns
    -------------
    nuclei_object
        inferred nuclei
    
    """
    min_hole_w = 0
    max_hole_w = 30
    small_obj_w = 0
    fill_filter_method = "3D"

    return infer_cellmask_fromcytoplasm(cytoplasm_mask,
                                         nucleus_mask,
                                         min_hole_w,
                                         max_hole_w,
                                         small_obj_w,
                                         fill_filter_method)




def infer_and_export_cellmask(
    in_img: np.ndarray, nuclei_obj: np.ndarray, meta_dict: Dict, out_data_path: Path
) -> np.ndarray:
    """
    infer cellmask and write inferred cellmask to ome.tif file

    Parameters
    ------------
    in_img:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    nuclei_obj:
        a 3d image containing the inferred nuclei
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """
    cellmask = fixed_infer_cellmask_fromcomposite(in_img, nuclei_obj)
    out_file_n = export_inferred_organelle(cellmask, "cell", meta_dict, out_data_path)
    print(f"inferred cellmask. wrote {out_file_n}")
    return cellmask>0


def get_cellmask(in_img: np.ndarray, nuclei_obj: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    load cellmask if it exists, otherwise calculate and write inferred cellmask to ome.tif file

    Parameters
    ------------
    in_img:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    nuclei_obj:
        a 3d image containing the inferred nuclei
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """

    try:
        cellmask = import_inferred_organelle("cell", meta_dict, out_data_path)
    except:
        start = time.time()
        print("starting segmentation...")
        cellmask = fixed_infer_cellmask_fromcomposite(in_img, nuclei_obj)
        out_file_n = export_inferred_organelle(cellmask, "cell", meta_dict, out_data_path)
        end = time.time()
        print(f"inferred (and exported) cellmask in ({(end - start):0.2f}) sec")

    return cellmask

##########################
#  create the cell mask by adding the nuclei and cytoplasm masks together
##########################
def combine_cytoplasm_and_nuclei(cyto_seg: np.ndarray,
                                 nuc_seg: np.ndarray,
                                 max_hole_width: int):
    """
    Function to combine the the cytoplasm and nuclei segmentations to produce the entire cell mask.

    Parameters:
    ----------
    cyto_seg: np.ndarray,
        image containing the cytoplasm segmentation
    nuc_seg: np.ndarray,
        image containing the nuclei segmentation
    max_hole_width: int
        size of the gaps between the nuclei and cytoplasm (usually small)
    """ 
    
    cells = np.logical_or(cyto_seg.astype(bool), nuc_seg.astype(bool))

    cell_multiple = fill_and_filter_linear_size(cells, 
                                                hole_min=0,
                                                hole_max=max_hole_width,
                                                min_size=0,
                                                method='3D')
    
    cell_area = cell_multiple.astype(bool)

    return cell_area


def select_highest_intensity_cell(raw_image: np.ndarray,
                                   cell_seg: np.ndarray,
                                   nuc_seg: np.ndarray):
    """ 
    Create an instance segmentation of the cell area using a watershed operation based on nuclei seeds.
    Then, select the cell with the highest combined organelle intensity.

    Parameters:
    ----------
    raw_image: np.ndarray,
        gray scale 3D multi-channel numpy array (CZYX)
    cell_seg: np.ndarray,
        binary cell segmentation with multiple cells in the FOV
    nuc_seg: np.ndarray,
        labeled nuclei segmentation with each nuclei having a different ID number (e.g., the result of the skimage label() function)
    labels_to_consider: Union(list, None)
        a list of labels that should be considered when determining the highest intensity. Default is None which utilizes all possible labels in the cell image
        
    Output
    ----------
    good_cell: np.ndarray  
        a binary image of the single cell with the highest total fluorescence intensity
    """
    # instance segmentation of cell area with watershed function
    cell_labels = masked_inverted_watershed(cell_seg, markers=nuc_seg, mask=cell_seg, method='3D')

    # create composite of all fluorescence channels after min-max normalization
    norm_channels = [(min_max_intensity_normalization(raw_image[c])) for c in range(len(raw_image))]
    normed_signal = np.stack(norm_channels, axis=0)
    normed_composite = normed_signal.sum(axis=0)

    # list of cell IDs to measure intensity of
    all_labels = np.unique(cell_labels)[1:]

    # measure total intensity in each cell from the ID list
    total_signal = [normed_composite[cell_labels == label].sum() for label in all_labels]

    # select the cell with the highest total intensity
    keep_label = all_labels[np.argmax(total_signal)]
    good_cell = cell_labels == keep_label

    return good_cell
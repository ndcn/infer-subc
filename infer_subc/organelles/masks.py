from typing import Union

import numpy as np

from infer_subc.core.img import *
from infer_subc.organelles.cellmask import (infer_cellmask_fromcomposite,
                                            infer_cellmask_fromcytoplasm,
                                            select_highest_intensity_cell,
                                            combine_cytoplasm_and_nuclei)

from infer_subc.organelles.nuclei import infer_nuclei_fromlabel, infer_nuclei_fromcytoplasm, mask_cytoplasm_nuclei, segment_nuclei_seeds
from infer_subc.organelles.cytoplasm import infer_cytoplasm, infer_cytoplasm_fromcomposite, infer_cytoplasm_fromcomposite, segment_cytoplasm_area

##################
# Masks Workflow #
##################
def infer_masks(in_img: np.ndarray, 
                nuc_ch: Union[int,None],
                nuc_median_sz: int, 
                nuc_gauss_sig: float,
                nuc_thresh_factor: float,
                nuc_thresh_min: float,
                nuc_thresh_max: float,
                nuc_min_hole_w: int,
                nuc_max_hole_w: int,
                nuc_small_obj_w: int,
                nuc_fill_filter_method: str,
                cell_weights: list[int],
                cell_rescale: bool,
                cell_median_sz: int,
                cell_gauss_sig: float,
                cell_mo_method: str,
                cell_mo_adjust: float,
                cell_mo_cutoff_size: int,
                cell_min_hole_w: int,
                cell_max_hole_w: int,
                cell_small_obj_w: int,
                cell_fill_filter_method: str,
                cell_watershed_method: str,
                cyto_erode_nuclei = True
                
                ):
    """
    Procedure to infer nucleus, cellmask, and cytoplams from multichannel confocal microscopy input image.

    Parameters
    ------------
    in_img: np.ndarray
        a 3d image containing all the organelle and nuclei label channels
    nuc_median_sz: int
        width of median filter for nuclei channel
    nuc_gauss_sig: float
        sigma for gaussian smoothing for nuclei channel
    nuc_thresh_factor: float
        adjustment factor for log Li threholding for nuclei channel
    nuc_thresh_min: float
        abs min threhold for log Li threholding for nuclei channel
    nuc_thresh_max: float
        abs max threhold for log Li threholding for nuclei channel
    nuc_max_hole_w: int
        hole filling cutoff for nuclei post-processing for nuclei channel
    nuc_small_obj_w: int
        minimum object size cutoff for nuclei post-processing for nuclei channel
    cell_weights:
        a list of int that corresond to the weights for each channel in the composite; use 0 if a channel should not be included in the composite image
    cell_rescale:
        True - rescale composite image
        False - don't rescale composite image
    cell_nuclei_labels: 
        a 3d image containing the inferred nuclei labels
    cell_median_sz: 
        width of median filter for _cellmask_ signal
    cell_gauss_sig: 
        sigma for gaussian smoothing of _cellmask_ signal
    cell_mo_method: 
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
    cell_mo_adjust: 
        Masked Object threshold `local_adjust`
    cell_mo_cutoff_size: 
        Masked Object threshold `size_min`
    cell_max_hole_w: 
        hole filling cutoff for cellmask signal post-processing
    cell_small_obj_w: 
        minimum object size cutoff for cellmask signal post-processing
    cell_fill_filter_method:
        determines if the fill and filter function should be run 'sice-by-slice' or in '3D' 
    cell_watershed_method:
        determines if the watershed should be run 'sice-by-slice' or in '3D' 
    cyto_erode_nuclei: 
        should we erode? Default False
    
    Output
    ------
    stacked_masks: np.ndarray
        a multi-channel np.ndarry containing the nuclei, cell, and cytoplasm masks
    """

    ####################
    ### infer nuclei ###
    ####################
    nuc_objs = infer_nuclei_fromlabel(in_img,
                                      nuc_ch,
                                      nuc_median_sz,
                                      nuc_gauss_sig,
                                      nuc_thresh_factor,
                                      nuc_thresh_min, 
                                      nuc_thresh_max, 
                                      nuc_min_hole_w, 
                                      nuc_max_hole_w,
                                      nuc_small_obj_w, 
                                      nuc_fill_filter_method)
    
    ######################
    ### infer cellmask ###
    ######################
    cellmask_obj = infer_cellmask_fromcomposite(in_img,
                                                cell_weights,
                                                # cell_rescale,
                                                nuc_objs,
                                                cell_median_sz,
                                                cell_gauss_sig,
                                                cell_mo_method,
                                                cell_mo_adjust,
                                                cell_mo_cutoff_size,
                                                cell_min_hole_w,
                                                cell_max_hole_w,
                                                cell_small_obj_w,
                                                cell_fill_filter_method,
                                                cell_watershed_method)

    ######################
    ### infer cellmask ###
    ###################### 
    cyto_obj = infer_cytoplasm(nuc_objs, 
                               cellmask_obj,
                               erode_nuclei=cyto_erode_nuclei)
    
    #########################
    ### select single nuc ###
    #########################
    nuc_obj = apply_mask(nuc_objs, cellmask_obj)
    nuc_obj = label_bool_as_uint16(nuc_obj)
    
    ###################
    ### stack masks ###
    ###################
    maskstack = stack_masks(nuc_mask=nuc_obj, 
                            cellmask=cellmask_obj,
                            cyto_mask=cyto_obj)
    
    return maskstack

####################
# Masks A Workflow #
####################
def infer_masks_A(in_img: np.ndarray,
                    cyto_weights: list[int],
                    cyto_rescale: bool,
                    cyto_median_sz: int,
                    cyto_gauss_sig: float,
                    cyto_mo_method: str,
                    cyto_mo_adjust: float,
                    cyto_mo_cutoff_size: int,
                    cyto_min_hole_w: int,
                    cyto_max_hole_w: int,
                    cyto_small_obj_w: int,
                    cyto_fill_filter_method: str,
                    nuc_min_hole_w: int,
                    nuc_max_hole_w: int,
                    nuc_fill_method: str,
                    nuc_small_obj_w: int,
                    nuc_fill_filter_method: str,
                    cell_min_hole_width: int,
                    cell_max_hole_width: int,
                    cell_small_obj_width: int,
                    cell_fill_filter_method: str
                    ) -> np.ndarray:
    """
    Procedure to infer cellmask from linear unmixed input.

    Parameters
    ------------
    in_img: 
        a 3d image containing all the channels
    cyto_weights:
        a list of int that corresond to the weights for each channel in the composite; use 0 if a channel should not be included in the composite image
    cyto_rescale:
        True = rescale composite
        False = don't rescale composite
    cyto_median_sz: 
        width of median filter for _cellmask_ signal
    cyto_gauss_sig: 
        sigma for gaussian smoothing of _cellmask_ signal
    cyto_mo_method: 
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
    cyto_mo_adjust: 
        Masked Object threshold `local_adjust`
    cyto_mo_cutoff_size: 
        Masked Object threshold `size_min`
    cyto_max_hole_w: 
        hole filling cutoff for cellmask signal post-processing
    cyto_small_obj_w: 
        minimum object size cutoff for cellmask signal post-processing
    cyto_fill_filter_method:
        determines if fill and filter should be run 'sice-by-slice' or in '3D' 
    nuc_max_hole_w: int
        hole filling cutoff to fill the nuclei
    nuc_small_obj_w: int
        object size cutoff to remove artifacts from dilation/erosion steps
    nuc_fill_filter_method: str
        to filter artifacts in "3D" or "slice-by-slice"
    cell_min_hole_width: int
        minimum size of holes to fill in final cell mask
    cell_max_hole_width: int,
        maximum size of holes to fill in final cell mask
    cell_small_obj_w: int
        minimum object size cutoff to remove from final cell mask; likely not required since small objects were removed from cytoplasm mask
    cell_fill_method: str
        method for fill and filter; either "3D" or "slice_by_slice"

    Returns
    -------------
    mask_stack:
        a logical/labels object defining boundaries of nucleus and cell mask

    """
    
    cyto_obj = infer_cytoplasm_fromcomposite(in_img,
                                            cyto_weights,
                                            # cyto_rescale,
                                            cyto_median_sz,
                                            cyto_gauss_sig,
                                            cyto_mo_method,
                                            cyto_mo_adjust,
                                            cyto_mo_cutoff_size,
                                            cyto_min_hole_w,
                                            cyto_max_hole_w,
                                            cyto_small_obj_w,
                                            cyto_fill_filter_method) 
    
    nuc_obj = infer_nuclei_fromcytoplasm(cyto_obj, 
                                         nuc_min_hole_w,
                                         nuc_max_hole_w,
                                         nuc_fill_method, 
                                         nuc_small_obj_w,)
                                        #  nuc_fill_filter_method)
    
    cell_obj = infer_cellmask_fromcytoplasm(cytoplasm_mask = cyto_obj, 
                                            nucleus_mask = nuc_obj,
                                            min_hole_width = cell_min_hole_width,
                                            max_hole_width = cell_max_hole_width,
                                            small_obj_width = cell_small_obj_width,
                                            fill_filter_method = cell_fill_filter_method)
    
    stack = stack_masks(nuc_mask=nuc_obj, cellmask=cell_obj, cyto_mask=cyto_obj)

    return stack

####################
# Masks B Workflow #
####################
def infer_masks_B(in_img: np.ndarray,
                   cyto_weights: list[int],
                   cyto_rescale: bool,
                   cyto_median_sz: int,
                   cyto_gauss_sig: float,
                   cyto_mo_method: str,
                   cyto_mo_adjust: float,
                   cyto_mo_cutoff_size: int,
                   cyto_min_hole_w: int,
                   cyto_max_hole_w: int,
                   cyto_small_obj_w: int,
                   cyto_fill_filter_method: str,
                   max_nuclei_width: int,
                   nuc_small_obj_width: int,
                   cell_fillhole_max: int,
                   cyto_small_object_width2: int) -> np.ndarray:
    
    """
    Procedure to infer cellmask from linear unmixed input.

    Parameters
    ------------
    in_img: 
        a 3d image containing all the channels
    weights:
        a list of int that corresond to the weights for each channel in the composite; use 0 if a channel should not be included in the composite image
    rescale:
        True = rescale composite
        False = don't rescale composite
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
    max_hole_w: 
        hole filling cutoff for cellmask signal post-processing
    small_obj_w: 
        minimum object size cutoff for cellmask signal post-processing
    fill_filter_method:
        determines if fill and filter should be run 'sice-by-slice' or in '3D' 
    cyto_fillhole_max: int
        size of the gaps between the nuclei and cytoplasm (usually small)
    max_nuclei_width: int
        the maximum expected width of the nuclei
    nuc_small_obj_width: int
        width of the any small objects that may be left over after nuclei are selected (i.e., errors in the seg)
    cell_fillhole_max: int
        size of the gaps between the nuclei and cytoplasm (usually small)
    cyto_small_object_width2: int
        size of the gaps between the nuclei and cytoplasm (usually small)

    Returns
    -------------
    mask_stack:
        a three channel np.ndarray constisting of the nucleus and cell (one object per channel)

    """
    cytoplasms = infer_cytoplasm_fromcomposite(in_img, 
                                       cyto_weights,
                                    #    cyto_rescale,
                                       cyto_median_sz,
                                       cyto_gauss_sig,
                                       cyto_mo_method,
                                       cyto_mo_adjust,
                                       cyto_mo_cutoff_size,
                                       cyto_min_hole_w,
                                       cyto_max_hole_w,
                                       cyto_small_obj_w,
                                       cyto_fill_filter_method)
    
    nuclei_seeds = segment_nuclei_seeds(cytoplasms, 
                              max_nuclei_width, 
                              nuc_small_obj_width)
    
    cellmasks = combine_cytoplasm_and_nuclei(cytoplasms, nuclei_seeds, cell_fillhole_max)
    
    good_CM = select_highest_intensity_cell(in_img, cellmasks, nuclei_seeds)
    
    good_nuc, good_cyto = mask_cytoplasm_nuclei(good_CM, cytoplasms, cyto_small_object_width2)
    
    stack = stack_masks(nuc_mask=good_nuc, cellmask=good_CM, cyto_mask=good_cyto)

    return stack
"""
infer_subc base module.

This is the principal module of the infer_subc project.
here you put your main classes and objects.
"""
import numpy as np
    
from platform import system
from collections import defaultdict

import scipy
from scipy import ndimage as ndi
from scipy.ndimage import median_filter, extrema
from scipy.interpolate import RectBivariateSpline

from skimage import img_as_float, filters
from skimage import morphology
from skimage.morphology import remove_small_objects, ball, disk, dilation, binary_closing, white_tophat, black_tophat
from skimage.filters import (
    threshold_triangle, 
    threshold_otsu, 
    threshold_li,
    threshold_multiotsu)
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from aicsimageio.writers import OmeTiffWriter
from napari_aicsimageio.core import  reader_function

import aicssegmentation
from aicssegmentation.core.seg_dot import (dot_3d_wrapper, 
                                                                            dot_slice_by_slice, 
                                                                            dot_2d_slice_by_slice_wrapper, 
                                                                            dot_3d)
from aicssegmentation.core.pre_processing_utils import ( intensity_normalization, 
                                                         image_smoothing_gaussian_3d,  
                                                         image_smoothing_gaussian_slice_by_slice,
                                                         edge_preserving_smoothing_3d )
from aicssegmentation.core.utils import (topology_preserving_thinning, 
                                                                    hole_filling, 
                                                                    size_filter)
from aicssegmentation.core.MO_threshold import MO
from aicssegmentation.core.vessel import filament_2d_wrapper, vesselnessSliceBySlice
from aicssegmentation.core.output_utils import   save_segmentation,  generate_segmentation_contour
                                                 


# example constant variable
NAME = "infer_subc"


# ## we need to define some image processing wrappers... partials should work great
# from functools import partial

# # derived from CellProfiler not sure if they work in 2D...
# def enhance_speckles(image, radius, volumetric = False):
#     if volumetric:
#         selem = ball(radius)
#     else:
#         selem = disk(radius)     
#     retval = white_tophat(image, footprint=selem)
#     return retval

# # derived from CellProfiler
# def enhance_neurites(image, radius, volumetric = False):
#     if volumetric:
#         selem = ball(radius)
#     else:
#         selem = disk(radius)     
#     white = white_tophat(image, footprint=selem)
#     black = black_tophat(image, footprint=selem)
#     result = image + white - black
#     result[result > 1] = 1
#     result[result < 0] = 0
#     return result

# these are the fuinctions that need to be set by each notebook...
# notebook workflow will produce the in_params dictionary nescessary 
# so all the images can be pushed through these functions (procedures)

##########################
# 1.  infer_NUCLEI
##########################
# copy this to base.py for easy import

def infer_NUCLEI(struct_img, in_params) -> tuple:
    """
    Procedure to infer NUCLEI from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the NUCLEI signal

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of NU
        label
            label (could be more than 1)
        signal
            scaled/filtered (pre-processed) flourescence image
        parameters: dict
            updated parameters in case any needed were missing
    
    """

    out_p= in_params.copy()


    ###################
    # PRE_PROCESSING
    ###################                         

    #TODO: replace params below with the input params
    scaling_param =  [0]   
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param

    med_filter_size = 4   
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter_slice_by_slice( 
                                                                    struct_img,
                                                                    size=med_filter_size  )
    out_p["median_filter_size"] = med_filter_size 

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(   struct_img,
                                                                                                        sigma=gaussian_smoothing_sigma,
                                                                                                        truncate_range = gaussian_smoothing_truncate_range
                                                                                                    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma 
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    ###################
    # CORE_PROCESSING
    ###################

    struct_obj = struct_img > filters.threshold_li(struct_img)
    threshold_value_log = threshold_li_log(struct_img)

    threshold_factor = 0.9 #from cellProfiler
    thresh_min = .1
    thresh_max = 1.
    threshold = min( max(threshold_value_log*threshold_factor, thresh_min), thresh_max)
    out_p['threshold_factor'] = threshold_factor
    out_p['thresh_min'] = thresh_min
    out_p['thresh_max'] = thresh_max

    struct_obj = struct_img > threshold

    ###################
    # POST_PROCESSING
    ###################

    hole_width = 5  
    # # wrapper to remoce_small_objects
    struct_obj = morphology.remove_small_holes(struct_obj, hole_width ** 3 )
    out_p['hole_width'] = hole_width


    small_object_max = 5
    struct_obj = aicssegmentation.core.utils.size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
                                                            min_size= small_object_max**3, 
                                                         method = "slice_by_slice", #"3D", # 
                                                            connectivity=1)
    out_p['small_object_max'] = small_object_max


    retval = (struct_obj,  label(struct_obj), out_p)
    return retval



##########################
# 2a.  infer_SOMA1
##########################
def infer_SOMA1(struct_img: np.ndarray, NU_labels: np.ndarray,  in_params:dict) -> tuple:
    """
    Procedure to infer SOMA from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the SOMA signal

    NU_labels: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of SOMA
        label
            label (could be more than 1)
        parameters: dict
            updated parameters in case any needed were missing
    
    """
    out_p= in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################                         

    #TODO: replace params below with the input params
    scaling_param =  [0]   
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param

    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    # Linear-ish processing
    med_filter_size = 15   
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter_slice_by_slice(  struct_img,
                                                                            size=med_filter_size  )
    out_p["median_filter_size"] = med_filter_size 
    gaussian_smoothing_sigma = 1.
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(   struct_img,
                                                                                                        sigma=gaussian_smoothing_sigma,
                                                                                                        truncate_range = gaussian_smoothing_truncate_range
                                                                                                    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma 
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    # non-Linear processing
    log_img, d = log_transform( struct_img ) 
    log_img = intensity_normalization(  log_img,  scaling_param=[0] )

    struct_img = intensity_normalization(  filters.scharr(log_img),  scaling_param=[0] )  + log_img

    ###################
    # CORE_PROCESSING
    ###################
    local_adjust = 0.5
    low_level_min_size = 100
    # "Masked Object Thresholding" - 3D
    struct_obj, _bw_low_level = MO(struct_img, 
                                                global_thresh_method='ave', 
                                                object_minArea=low_level_min_size, 
                                                extra_criteria=True,
                                                local_adjust= local_adjust, 
                                                return_object=True,
                                                dilate=True)

    out_p["local_adjust"] = local_adjust 
    out_p["low_level_min_size"] = low_level_min_size 

    ###################
    # POST_PROCESSING
    ###################

    # 2D 
    hole_max = 80  
    struct_obj = hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    out_p['hole_max'] = hole_max

    small_object_max = 35
    struct_obj = size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
                                                            min_size= small_object_max**3, 
                                                            method = "slice_by_slice" ,
                                                            connectivity=1)
    out_p['small_object_max'] = small_object_max

    labels_out = watershed(
                connectivity=np.ones((3, 3,3), bool),
                image=1. - struct_img,
                markers=NU_labels,
                mask= np.logical_or(struct_obj, NU_labels > 0),
                )
    ###################
    # POST- POST_PROCESSING
    ###################
    # keep the "SOMA" label which contains the highest total signal
    all_labels = np.unique(labels_out)[1:]

    total_signal = [ scaled_signal[labels_out == label].sum() for label in all_labels]
    # combine NU and "labels" to make a SOMA
    keep_label = all_labels[np.argmax(total_signal)]

    # now use all the NU labels which AREN't keep_label and add to mask and re-label
    masked_composite_soma = struct_img.copy()
    new_NU_mask = np.logical_and( NU_labels !=0 ,NU_labels != keep_label)

    # "Masked Object Thresholding" - 3D
    masked_composite_soma[new_NU_mask] = 0
    struct_obj, _bw_low_level = MO(masked_composite_soma, 
                                                global_thresh_method='ave', 
                                                object_minArea=low_level_min_size, 
                                                extra_criteria=True,
                                                local_adjust= local_adjust, 
                                                return_object=True,
                                                dilate=True)

    struct_obj = hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    struct_obj = size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
                                                            min_size= small_object_max**3, 
                                                            method = "slice_by_slice" ,
                                                            connectivity=1)
    masked_labels_out = watershed(
                connectivity=np.ones((3, 3,3), bool),
                image=1. - struct_img,
                markers=NU_labels,
                mask= np.logical_or(struct_obj, NU_labels == keep_label),
                )
                

    retval = (struct_obj,  masked_labels_out, out_p)
    return retval




##########################
# 2b.  infer_SOMA2
########################### copy this to base.py for easy import
def infer_SOMA2(struct_img: np.ndarray, NU_labels: np.ndarray,  in_params:dict) -> tuple:
    """
    Procedure to infer SOMA from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the SOMA signal

    NU_labels: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of SOMA
        label
            label (could be more than 1)
        parameters: dict
            updated parameters in case any needed were missing
    
    """
    out_p= in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################                         
    #TODO: replace params below with the input params
    scaling_param =  [0]   
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param


    # 2D smoothing
    # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    med_filter_size = 9   
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter_slice_by_slice( 
                                                                    struct_img,
                                                                    size=med_filter_size  )
    out_p["median_filter_size"] = med_filter_size 

    gaussian_smoothing_sigma = 3.
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(   struct_img,
                                                                                                        sigma=gaussian_smoothing_sigma,
                                                                                                        truncate_range = gaussian_smoothing_truncate_range
                                                                                                    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma 
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    #    edges = filters.scharr(struct_img)
    # struct_img, d = log_transform( struct_img ) 
    # struct_img = intensity_normalization(  struct_img,  scaling_param=[0] )
    ###################
    # CORE_PROCESSING
    ###################
    # "Masked Object Thresholding" - 3D
    local_adjust = 0.25
    low_level_min_size = 100
    struct_obj, _bw_low_level = MO(struct_img, 
                                                global_thresh_method='ave', 
                                                object_minArea=low_level_min_size, 
                                                extra_criteria=True,
                                                local_adjust= local_adjust, 
                                                return_object=True,
                                                dilate=True)
    out_p["local_adjust"] = local_adjust 
    out_p["low_level_min_size"] = low_level_min_size 

    ###################
    # POST_PROCESSING
    ###################
    # 3D cleaning

    hole_max = 80  
    # discount z direction
    struct_obj = hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    out_p['hole_max'] = hole_max

    small_object_max = 35
    struct_obj = size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
                                                            min_size= small_object_max**3, 
                                                            method = "3D", #"slice_by_slice" 
                                                            connectivity=1)
    out_p['small_object_max'] = small_object_max

    labels_out = watershed(
                                                image=np.abs(ndi.sobel(struct_img)),
                                                markers=NU_labels,
                                                connectivity=np.ones((3, 3, 3), bool),
                                                mask= np.logical_or(struct_obj, NU_labels > 0),
                                                )

    ###################
    # POST- POST_PROCESSING
    ###################
    # keep the "SOMA" label which contains the highest total signal
    all_labels = np.unique(labels_out)[1:]

    total_signal = [ scaled_signal[labels_out == label].sum() for label in all_labels]
    # combine NU and "labels" to make a SOMA
    keep_label = all_labels[np.argmax(total_signal)]

    # now use all the NU labels which AREN't keep_label and add to mask and re-label
    masked_composite_soma = struct_img.copy()
    new_NU_mask = np.logical_and( NU_labels !=0 ,NU_labels != keep_label)

    # "Masked Object Thresholding" - 3D
    masked_composite_soma[new_NU_mask] = 0
    struct_obj, _bw_low_level = MO(masked_composite_soma, 
                                                global_thresh_method='ave', 
                                                object_minArea=low_level_min_size, 
                                                extra_criteria=True,
                                                local_adjust= local_adjust, 
                                                return_object=True,
                                                dilate=True)
    # 3D cleaning
    struct_obj = hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    struct_obj = size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
                                                            min_size= small_object_max**3, 
                                                            method = "3D", #"slice_by_slice" 
                                                            connectivity=1)
    masked_labels_out = watershed(
                connectivity=np.ones((3, 3,3), bool),
                image=np.abs(ndi.sobel(struct_img)),
                markers=NU_labels,
                mask= np.logical_or(struct_obj, NU_labels == keep_label),
                )
                

    retval = (struct_obj,  masked_labels_out, out_p)
    return retval

##########################
# 2c.  infer_SOMA3
##########################
def infer_SOMA3(struct_img, NU_labels,  in_params) -> tuple:
    """
    Procedure to infer SOMA from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the SOMA signal

    NU_labels: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of SOMA
        label
            label (could be more than 1)
        parameters: dict
            updated parameters in case any needed were missing
    
    """
    out_p= in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################                         
    scaling_param =  [0]   
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param

   # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    med_filter_size = 3   
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter_slice_by_slice( 
                                                                    struct_img,
                                                                    size=med_filter_size  )
    out_p["median_filter_size"] = med_filter_size 

    gaussian_smoothing_sigma = 1.
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(   struct_img,
                                                                                                        sigma=gaussian_smoothing_sigma,
                                                                                                        truncate_range = gaussian_smoothing_truncate_range
                                                                                                    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma 
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    log_img, d = log_transform( struct_img ) 
    struct_img = intensity_normalization(  log_img + filters.scharr(log_img) ,  scaling_param=[0] )  


    ###################
    # CORE_PROCESSING
    ###################
    # "Masked Object Thresholding" - 3D
    local_adjust = 0.5
    low_level_min_size = 100
    struct_obj, _bw_low_level = MO(struct_img, 
                                                global_thresh_method='ave', 
                                                object_minArea=low_level_min_size, 
                                                extra_criteria=True,
                                                local_adjust= local_adjust, 
                                                return_object=True,
                                                dilate=True)
    out_p["local_adjust"] = local_adjust 
    out_p["low_level_min_size"] = low_level_min_size 
    ###################
    # POST_PROCESSING
    ###################
    # 2D cleaning
    hole_max = 100  
    # discount z direction
    struct_obj = hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    out_p['hole_max'] = hole_max

    small_object_max = 30
    struct_obj = size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
                                                            min_size= small_object_max**3, 
                                                            method = "slice_by_slice" ,
                                                            connectivity=1)
    out_p['small_object_max'] = small_object_max

    labels_out = watershed(
                                                image=np.abs(ndi.sobel(struct_img)),  #either log_img or struct_img seem to work, but more spurious labeling to fix in post-post for struct_img
                                                markers=NU_labels,
                                                connectivity=np.ones((3, 3, 3), bool),
                                                mask= np.logical_or(struct_obj, NU_labels > 0),
                                                )

    ###################
    # POST- POST_PROCESSING
    ###################
    # keep the "SOMA" label which contains the highest total signal
    all_labels = np.unique(labels_out)[1:]

    total_signal = [ scaled_signal[labels_out == label].sum() for label in all_labels]
    # combine NU and "labels" to make a SOMA
    keep_label = all_labels[np.argmax(total_signal)]

    # now use all the NU labels which AREN't keep_label and add to mask and re-label
    masked_composite_soma = struct_img.copy()
    new_NU_mask = np.logical_and( NU_labels !=0 ,NU_labels != keep_label)

    # "Masked Object Thresholding" - 3D
    masked_composite_soma[new_NU_mask] = 0
    struct_obj, _bw_low_level = MO(masked_composite_soma, 
                                                global_thresh_method='ave', 
                                                object_minArea=low_level_min_size, 
                                                extra_criteria=True,
                                                local_adjust= local_adjust, 
                                                return_object=True,
                                                dilate=True)

    # 2D cleaning
    struct_obj = hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    struct_obj = size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
                                                            min_size= small_object_max**3, 
                                                            method = "slice_by_slice" ,
                                                            connectivity=1)
    masked_labels_out = watershed(
                connectivity=np.ones((3, 3,3), bool),
                image=np.abs(ndi.sobel(struct_img)),  #either log_img or struct_img seem to work, but more spurious labeling to fix in post-post for struct_img
                markers=NU_labels,
                mask= np.logical_or(struct_obj, NU_labels == keep_label),
                )
                
    retval = (struct_obj,  masked_labels_out, out_p)
    return retval


##########################
#  infer_CYTOSOL
##########################
def infer_CYTOSOL(SO_object, NU_object, erode_NU = True):
    """
    Procedure to infer CYTOSOL from linearly unmixed input.

    Parameters:
    ------------
    SO_object: np.ndarray
        a 3d image containing the NUCLEI signal

    NU_object: np.ndarray
        a 3d image containing the NUCLEI signal

    erode_NU: bool
        should we erode?

    Returns:
    -------------
    CY_object: np.ndarray (bool)
      
    """

    #NU_eroded1 = morphology.binary_erosion(NU_object,  footprint=morphology.ball(3) )
    if erode_NU:
        CY_object = np.logical_and(SO_object,~morphology.binary_erosion(NU_object) )
    else:
        CY_object = np.logical_and(SO_object,~NU_object)
    return CY_object


##########################
#  infer_LYSOSOMES
##########################
def infer_LYSOSOMES(struct_img,  CY_object,  in_params) -> tuple:
    """
    Procedure to infer LYSOSOME from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the LYSOSOME signal

    CY_object: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of Lysosomes
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p= in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################                         
    scaling_param =  [0]   
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param

   # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    med_filter_size = 3   
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter_slice_by_slice( 
                                                                    struct_img,
                                                                    size=med_filter_size  )
    out_p["median_filter_size"] = med_filter_size 

    gaussian_smoothing_sigma = 1.
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(   struct_img,
                                                                                                        sigma=gaussian_smoothing_sigma,
                                                                                                        truncate_range = gaussian_smoothing_truncate_range
                                                                                                    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma 
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    # log_img, d = log_transform( struct_img ) 
    # struct_img = intensity_normalization( log_img ,  scaling_param=[0] )  
    struct_img = intensity_normalization( struct_img ,  scaling_param=[0] )  


   ###################
    # CORE_PROCESSING
    ###################
    # dot and filiment enhancement - 2D

    ################################
    ## PARAMETERS for this step ##
    s2_param = [[5,0.09], [2.5,0.07], [1,0.01]]
    ################################
    bw_spot = dot_2d_slice_by_slice_wrapper(struct_img, s2_param)

    ################################
    ## PARAMETERS for this step ##
    f2_param = [[1, 0.15]]
    ################################
    bw_filament = filament_2d_wrapper(struct_img, f2_param)

    bw = np.logical_or(bw_spot, bw_filament)

    out_p["s2_param"] = s2_param 
    out_p["f2_param"] = f2_param 
    ###################
    # POST_PROCESSING
    ###################

    # 2D cleaning
    hole_max = 1600  
    # discount z direction
    struct_obj = aicssegmentation.core.utils.hole_filling(bw, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    out_p['hole_max'] = hole_max


    # # 3D
    # cleaned_img = remove_small_objects(removed_holes>0, 
    #                                                             min_size=width, 
    #                                                             connectivity=1, 
    #                                                             in_place=False)

    small_object_max = 15
    struct_obj = aicssegmentation.core.utils.size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
                                                            min_size= small_object_max**3, 
                                                            method = "3D", #"slice_by_slice" ,
                                                            connectivity=1)
    out_p['small_object_max'] = small_object_max

                
    retval = (struct_obj,  out_p)
    return retval


##########################
#  infer_MITOCHONDRIA
##########################
def infer_MITOCHONDRIA(struct_img, CY_object,  in_params) -> tuple:
    """
    Procedure to infer MITOCHONDRIA  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the MITOCHONDRIA signal

    CY_object: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of MITOCHONDRIA
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p= in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################                         
    scaling_param =  [0,9]   
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param

   # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    med_filter_size = 3   
    structure_img = median_filter(struct_img,    size=med_filter_size  )
    out_p["median_filter_size"] = med_filter_size 

    gaussian_smoothing_sigma = 1.3
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_3d(   struct_img,
                                                                                                        sigma=gaussian_smoothing_sigma,
                                                                                                        truncate_range = gaussian_smoothing_truncate_range
                                                                                                    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma 
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    # log_img, d = log_transform( struct_img ) 
    # struct_img = intensity_normalization( log_img ,  scaling_param=[0] )  
    # struct_img = intensity_normalization( struct_img ,  scaling_param=[0] )  


   ###################
    # CORE_PROCESSING
    ###################
    ################################
    ## PARAMETERS for this step ##
    vesselness_sigma = [1.5]
    vesselness_cutoff = 0.16
    # 2d vesselness slice by slice
    response = vesselnessSliceBySlice(struct_img, sigmas=vesselness_sigma, tau=1, whiteonblack=True)
    bw = response > vesselness_cutoff

    out_p["vesselness_sigma"] = vesselness_sigma 
    out_p["vesselness_cutoff"] = vesselness_cutoff 


    ###################
    # POST_PROCESSING
    ###################

    # MT_object = remove_small_objects(bw > 0, min_size=small_object_max**2, connectivity=1, in_place=False)
    small_object_max = 10
    struct_obj = size_filter(bw, # wrapper to remove_small_objects which can do slice by slice
                                                            min_size= small_object_max**3, 
                                                            method = "3D", #"slice_by_slice" ,
                                                            connectivity=1)
    out_p['small_object_max'] = small_object_max

    retval = (struct_obj,  out_p)
    return retval


##########################
#  infer_GOLGI
##########################
def infer_GOLGI(struct_img, CY_object,  in_params) -> tuple:
    """
    Procedure to infer GOLGI COMPLEX  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the GOLGI signal

    CY_object: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of GOLGI
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p= in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################                         
    intensity_norm_param = [0.1, 30.]  # CHECK THIS

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    out_p["intensity_norm_param"] = intensity_norm_param

   # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    med_filter_size = 3   
    structure_img = median_filter(struct_img,    size=med_filter_size  )
    out_p["median_filter_size"] = med_filter_size 

    gaussian_smoothing_sigma = 1.
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_3d(   struct_img,
                                                                                                        sigma=gaussian_smoothing_sigma,
                                                                                                        truncate_range = gaussian_smoothing_truncate_range
                                                                                                    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma 
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    # log_img, d = log_transform( struct_img ) 
    # struct_img = intensity_normalization( log_img ,  scaling_param=[0] )  
    # struct_img = intensity_normalization( struct_img ,  scaling_param=[0] )  


   ###################
    # CORE_PROCESSING
    ###################
    cell_wise_min_area = 1200
    bw, object_for_debug = MO(struct_img, 
                                                global_thresh_method='tri', 
                                                object_minArea=cell_wise_min_area, 
                                                return_object=True)
    out_p["cell_wise_min_area"] = cell_wise_min_area 

    thin_dist_preserve=1.6
    thin_dist=1
    bw_thin = topology_preserving_thinning(bw>0, thin_dist_preserve, thin_dist)
    out_p["thin_dist_preserve"] = thin_dist_preserve 
    out_p["thin_dist"] = thin_dist 

    dot_3d_sigma = 1.6
    dot_3d_cutoff = 0.02
    s3_param = [(dot_3d_sigma,dot_3d_cutoff)]

    bw_extra = dot_3d_wrapper(struct_img, s3_param)
    out_p["dot_3d_sigma"] = dot_3d_sigma 
    out_p["dot_3d_cutoff"] = dot_3d_cutoff 
    out_p["s3_param"] = s3_param 

    bw = np.logical_or(bw_extra>0, bw_thin)


    ###################
    # POST_PROCESSING
    ###################

    # MT_object = remove_small_objects(bw > 0, min_size=small_object_max**2, connectivity=1, in_place=False)
    small_object_max = 10
    struct_obj = size_filter(bw, # wrapper to remove_small_objects which can do slice by slice
                                                            min_size= small_object_max**3, 
                                                            method = "3D", #"slice_by_slice" ,
                                                            connectivity=1)
    out_p['small_object_max'] = small_object_max


    retval = (struct_obj,  out_p)
    return retval


##########################
#  infer_PEROXISOME
##########################
def infer_PEROXISOME(struct_img, CY_object,  in_params) -> tuple:
    """
    Procedure to infer PEROXISOME  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the PEROXISOME signal

    CY_object: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of PEROXISOME
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p= in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################                         
    intensity_norm_param = [0]  # CHECK THIS

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    out_p["intensity_norm_param"] = intensity_norm_param

   # make a copy for post-post processing
    scaled_signal = struct_img.copy()

    gaussian_smoothing_sigma = 1.
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_3d(   struct_img,
                                                                                                        sigma=gaussian_smoothing_sigma,
                                                                                                        truncate_range = gaussian_smoothing_truncate_range
                                                                                                    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma 
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    # log_img, d = log_transform( struct_img ) 
    # struct_img = intensity_normalization( log_img ,  scaling_param=[0] )  
    # struct_img = intensity_normalization( struct_img ,  scaling_param=[0] )  


   ###################
    # CORE_PROCESSING
    ###################
    dot_3d_sigma = 1.
    dot_3d_cutoff = 0.04
    s3_param = [(dot_3d_sigma,dot_3d_cutoff)]

    bw = dot_3d_wrapper(struct_img, s3_param)
    out_p["dot_3d_sigma"] = dot_3d_sigma 
    out_p["dot_3d_cutoff"] = dot_3d_cutoff 
    out_p["s3_param"] = s3_param 


    ###################
    # POST_PROCESSING
    ###################
    # watershed
    minArea = 4
    mask_ = remove_small_objects(bw>0, min_size=minArea, connectivity=1, in_place=False) 
    seed_ = dilation(peak_local_max(struct_img,labels=label(mask_), min_distance=2, indices=False), selem=ball(1))
    watershed_map = -1*ndi.distance_transform_edt(bw)
    struct_obj = watershed(watershed_map, label(seed_), mask=mask_, watershed_line=True)
    ################################
    ## PARAMETERS for this step ##
    min_area = 4 
    ################################
    struct_obj = remove_small_objects(struct_obj>0, min_size=min_area, connectivity=1, in_place=False)
    out_p["min_area"] = min_area 



    retval = (struct_obj,  out_p)
    return retval


##########################
#  infer_ENDOPLASMIC_RETICULUM
##########################
def infer_ENDOPLASMIC_RETICULUM(struct_img, CY_object,  in_params) -> tuple:
    """
    Procedure to infer PEROXISOME  from linearly unmixed input.

    Parameters:
    ------------
    struct_img: np.ndarray
        a 3d image containing the PEROXISOME signal

    CY_object: np.ndarray boolean
        a 3d image containing the NU labels

    in_params: dict
        holds the needed parameters

    Returns:
    -------------
    tuple of:
        object
            mask defined boundaries of PEROXISOME
        parameters: dict
            updated parameters in case any needed were missing
    """
    out_p= in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################                         
    intensity_norm_param = [0]  # CHECK THIS

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    out_p["intensity_norm_param"] = intensity_norm_param

    # edge-preserving smoothing (Option 2, used for Sec61B)
    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)


   ###################
    # CORE_PROCESSING
    ###################
    ################################
    ## PARAMETERS for this step ##
    f2_param = [[1, 0.15]]
    ################################

    struct_obj = filament_2d_wrapper(struct_img, f2_param)
    out_p["f2_param"] = f2_param 

    ###################
    # POST_PROCESSING
    ###################
 
    ################################
    ## PARAMETERS for this step ##
    min_area = 20
    ################################
    struct_obj = remove_small_objects(struct_obj>0, min_size=min_area, connectivity=1, in_place=False)
    out_p["min_area"] = min_area 



    retval = (struct_obj,  out_p)
    return retval


def infer_LIPID_DROPLET(struct_img, out_path, cyto_labels, in_params):
    pass


#### Median FIltering for 2D
# TODO: rewrite these with np.vectorize or enumerate...
#           mask_labeled = np.vectorize(keep_top_3, signature='(n,m)->(n,m)')(mask_labeled)


# We need to define a wrapper for `median_filter` which steps through each Z-slice independently.  (Note: since we will use this 
# pattern repeatedly we may want to make a generic wrapper for our filtering/de-noising). Lets call it `median_filter_slice_by_slice` 
# and copy the way the `aicssegmentation` package handles smoothing.
# TODO: typehints.... what is my "image" primitive?
def median_filter_slice_by_slice(struct_img, size):
    """
    wrapper for applying 2D median filter slice by slice on a 3D image
    """
    structure_img_denoise = np.zeros_like(struct_img)
    for zz in range(struct_img.shape[0]):
        structure_img_denoise[zz, :, :] = median_filter(struct_img[zz, :, :], size=size) 
        
    return structure_img_denoise


def simple_intensity_normalization(struct_img, max_value=None):
    """Normalize the intensity of input image so that the value range is from 0 to 1.

    Parameters:
    ------------
    img: np.ndarray
        a 3d image
    max_value: float
        
    """
    if max_value is not None:
        struct_img[struct_img > max_value] = max_value

    strech_min = struct_img.min()
    strech_max = struct_img.max()

    struct_img = (struct_img - strech_min + 1e-8) / (strech_max - strech_min + 1e-8)

    return struct_img


def read_input_image(image_name):
    # from aicsimageio import AICSImage
    # from napari_aicsimageio.core import _get_full_image_data, _get_meta
    # img_in = AICSImage(czi_image_name)
    # data_out = _get_full_image_data(img_in, in_memory=True)
    # meta_out = _get_meta(data, img_in)
    # meta_out['AICSImage'] = img_in

    # prefer this wrapper because it returns numpy arrays
    # or more simply with napari_aicsimagie io
    data_out, meta_out, layer_type = reader_function(image_name)[0]
    return (data_out,meta_out)



def log_transform(image):
    """Renormalize image intensities to log space
    
    Returns a tuple of transformed image and a dictionary to be passed into
    inverse_log_transform. The minimum and maximum from the dictionary
    can be applied to an image by the inverse_log_transform to 
    convert it back to its former intensity values.
    """

    orig_min, orig_max = extrema(image)[:2]
    #
    # We add 1/2 bit noise to an 8 bit image to give the log a bottom
    #
    limage = image.copy()
    noise_min = orig_min + (orig_max - orig_min) / 256.0 + np.finfo(image.dtype).eps
    limage[limage < noise_min] = noise_min
    d = {"noise_min": noise_min}
    limage = np.log(limage)
    log_min, log_max = extrema(limage)[:2]
    d["log_min"] = log_min
    d["log_max"] = log_max
    return stretch(limage), d


def inverse_log_transform(image, d):
    """Convert the values in image back to the scale prior to log_transform
    
    image - an image or value or values similarly scaled to image
    d - object returned by log_transform
    """
    return np.exp(unstretch(image, d["log_min"], d["log_max"]))


def stretch(image, mask=None):
    """Normalize an image to make the minimum zero and maximum one
    image - pixel data to be normalized
    mask  - optional mask of relevant pixels. None = don't mask
    returns the stretched image
    """
    image = np.array(image, float)
    if np.product(image.shape) == 0:
        return image
    if mask is None:
        minval = np.min(image)
        maxval = np.max(image)
        if minval == maxval:
            if minval < 0:
                return np.zeros_like(image)
            elif minval > 1:
                return np.ones_like(image)
            return image
        else:
            return (image - minval) / (maxval - minval)
    else:
        significant_pixels = image[mask]
        if significant_pixels.size == 0:
            return image
        minval = np.min(significant_pixels)
        maxval = np.max(significant_pixels)
        if minval == maxval:
            transformed_image = minval
        else:
            transformed_image = (significant_pixels - minval) / (maxval - minval)
        result = image.copy()
        image[mask] = transformed_image
        return image


def unstretch(image, minval, maxval):
    """Perform the inverse of stretch, given a stretched image
    image - an image stretched by stretch or similarly scaled value or values
    minval - minimum of previously stretched image
    maxval - maximum of previously stretched image
    """
    return image * (maxval - minval) + minval


def threshold_li_log( image_in ):
    """
    thin wrapper to log-scale and inverse log image for threshold finding using li minimum cross-entropy
    """

    image, d = log_transform(image_in.copy())
    threshold =  threshold_li(image)
    threshold = inverse_log_transform(threshold, d)
    return threshold

    image_out = log_transform(image_in.copy())
    li_thresholded = structure_img_smooth >threshold_li_log( structure_img_smooth )

def threshold_otsu_log( image_in ):
    """
    thin wrapper to log-scale and inverse log image for threshold finding using otsu
    """

    image, d = log_transform(image_in.copy())
    threshold =  threshold_otsu(image)
    threshold = inverse_log_transform(threshold, d)
    return threshold

def threshold_multiotsu_log( image_in ):
    """
    thin wrapper to log-scale and inverse log image for threshold finding using otsu
    """
    image, d = log_transform(image_in.copy())
    thresholds =  threshold_multiotsu(image)
    thresholds = inverse_log_transform(thresholds, d)
    return thresholds







def export_ome_tiff(data_in, meta_in, img_name, out_path, curr_chan=0) ->  str:
    #  data_in: types.ArrayLike,
    #  meta_in: dict,
    # img_name: types.PathLike,
    # out_path: types.PathLike,
    # curr_chan: int
    # assumes a single image


    out_name = out_path + img_name + ".ome.tiff"
   
    image_names = [img_name]
    print(image_names)
    #chan_names = meta_in['metadata']['aicsimage'].channel_names

    physical_pixel_sizes = [ meta_in['metadata']['aicsimage'].physical_pixel_sizes ]

    dimension_order = ["CZYX"]
    channel_names= [ meta_in['metadata']['aicsimage'].channel_names[curr_chan] ]
    if len(data_in.shape) == 3: #single channel zstack
        data_in=data_in[np.newaxis,:,:,:]
    
    if data_in.dtype == 'bool':
        data_in = data_in.astype(np.uint8)
        data_in[ data_in > 0 ] = 255

    out_ome = OmeTiffWriter.build_ome(
                    [ data_in.shape],
                    [data_in.dtype],
                    channel_names=channel_names,  # type: ignore
                    image_name=image_names,
                    physical_pixel_sizes=physical_pixel_sizes,
                    dimension_order=dimension_order,
                )


    OmeTiffWriter.save( data_in,
                                        out_name,
                                        dim_order=dimension_order,
                                        channel_names = channel_names,
                                        image_names = image_names,
                                        physical_pixel_sizes = physical_pixel_sizes,
                                        ome_xml=out_ome,
                        )
    return out_name


### UTILS
def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d



def get_raw_meta_data(meta_dict):
    curr_platform = system()

    if curr_platform=='Linux':
        raw_meta_data = meta_dict['metadata']['raw_image_metadata'].dict()
        ome_types = meta_dict['metadata']['ome_types']
    elif curr_platform=='Darwin':
        raw_meta_data = meta_dict['metadata']['raw_image_metadata']
        ome_types = []
    else:
        raw_meta_data = meta_dict['metadata']['raw_image_metadata']
        ome_types = []
        print(f"warning: platform = '{curr_platform}' is untested")
    return (raw_meta_data,ome_types)





# takein from cellprofiler / centrosome
# since i'm not limiting myself to integers it might not work...
def fill_labeled_holes(labels, mask=None, size_fn=None):
    """Fill all background pixels that are holes inside the foreground
 
    A pixel is a hole inside a foreground object if
    
    * there is no path from the pixel to the edge AND
    
    * there is no path from the pixel to any other non-hole
      pixel AND
      
    * there is no path from the pixel to two similarly-labeled pixels that
      are adjacent to two differently labeled non-hole pixels.
    
    labels - the current labeling
    
    mask - mask of pixels to ignore
    
    size_fn - if not None, it is a function that takes a size and a boolean
              indicating whether it is foreground (True) or background (False)
              The function should return True to analyze and False to ignore
    
    returns a filled copy of the labels matrix
    """
    #
    # The algorithm:
    #
    # Label the background to get distinct background objects
    # Construct a graph of both foreground and background objects.
    # Walk the graph according to the rules.
    #
    labels_type = labels.dtype
    background = labels == 0
    if mask is not None:
        background &= mask

    blabels, count = ndi.label(background, four_connect)
    labels = labels.copy().astype(int)
    lcount = np.max(labels)
    labels[blabels != 0] = blabels[blabels != 0] + lcount + 1
    lmax = lcount + count + 1
    is_not_hole = np.ascontiguousarray(np.zeros(lmax + 1, np.uint8))
    #
    # Find the indexes on the edge and use to populate the to-do list
    #
    to_do = np.unique(
        np.hstack((labels[0, :], labels[:, 0], labels[-1, :], labels[:, -1]))
    )
    to_do = to_do[to_do != 0]
    is_not_hole[to_do] = True
    to_do = list(to_do)
    #
    # An array that names the first non-hole object
    #
    adjacent_non_hole = np.ascontiguousarray(np.zeros(lmax + 1), np.uint32)
    #
    # Find all 4-connected adjacent pixels
    # Note that there will be some i, j not in j, i
    #
    i = np.hstack([labels[:-1, :].flatten(), labels[:, :-1].flatten()])
    j = np.hstack([labels[1:, :].flatten(), labels[:, 1:].flatten()])
    i, j = i[i != j], j[i != j]
    if (len(i)) > 0:
        order = np.lexsort((j, i))
        i = i[order]
        j = j[order]
        # Remove duplicates and stack to guarantee that j, i is in i, j
        first = np.hstack([[True], (i[:-1] != i[1:]) | (j[:-1] != j[1:])])
        i, j = np.hstack((i[first], j[first])), np.hstack((j[first], i[first]))
        # Remove dupes again. (much shorter)
        order = np.lexsort((j, i))
        i = i[order]
        j = j[order]
        first = np.hstack([[True], (i[:-1] != i[1:]) | (j[:-1] != j[1:])])
        i, j = i[first], j[first]
        #
        # Now we make a ragged array of i and j
        #
        i_count = np.bincount(i)
        if len(i_count) < lmax + 1:
            i_count = np.hstack((i_count, np.zeros(lmax + 1 - len(i_count), int)))
        indexer = Indexes([i_count])
        #
        # Filter using the size function passed, if any
        #
        if size_fn is not None:
            areas = np.bincount(labels.flatten())
            for ii, area in enumerate(areas):
                if (
                    ii > 0
                    and area > 0
                    and not is_not_hole[ii]
                    and not size_fn(area, ii <= lcount)
                ):
                    is_not_hole[ii] = True
                    to_do.append(ii)

        to_do_count = len(to_do)
        if len(to_do) < len(is_not_hole):
            to_do += [0] * (len(is_not_hole) - len(to_do))
        to_do = np.ascontiguousarray(np.array(to_do), np.uint32)
        fill_labeled_holes_loop(
            np.ascontiguousarray(i, np.uint32),
            np.ascontiguousarray(j, np.uint32),
            np.ascontiguousarray(indexer.fwd_idx, np.uint32),
            np.ascontiguousarray(i_count, np.uint32),
            is_not_hole,
            adjacent_non_hole,
            to_do,
            lcount,
            to_do_count,
        )
    #
    # Make an array that assigns objects to themselves and background to 0
    #
    new_indexes = np.arange(len(is_not_hole)).astype(np.uint32)
    new_indexes[(lcount + 1) :] = 0
    #
    # Fill the holes by replacing the old value by the value of the
    # enclosing object.
    #
    is_not_hole = is_not_hole.astype(bool)
    new_indexes[~is_not_hole] = adjacent_non_hole[~is_not_hole]
    if mask is not None:
        labels[mask] = new_indexes[labels[mask]]
    else:
        labels = new_indexes[labels]
    return labels.astype(labels_type)


TS_GLOBAL = "Global"
TS_ADAPTIVE = "Adaptive"
TM_MANUAL = "Manual"
TM_MEASUREMENT = "Measurement"
TM_LI = "Minimum Cross-Entropy"
TM_OTSU = "Otsu"
TM_ROBUST_BACKGROUND = "Robust Background"
TM_SAUVOLA = "Sauvola"

def get_global_threshold(
                                image, 
                                mask, 
                                threshold_operation=TM_LI, 
                                automatic=False, 
                                two_class_otsu=False, 
                                assign_mid_to_fg = True
                                ):

    image_data = image[mask]
    # Shortcuts - Check if image array is empty or all pixels are the same value.
    if len(image_data) == 0:
        threshold = 0.0

    elif np.all(image_data == image_data[0]):
        threshold = image_data[0]

    elif automatic or threshold_operation in (TM_LI, TM_SAUVOLA):
        tol = max(np.min(np.diff(np.unique(image_data))) / 2, 0.5 / 65536)
        threshold = skimage.filters.threshold_li(image_data, tolerance=tol)

    elif threshold_operation == TM_OTSU:
        if two_class_otsu:
            threshold = skimage.filters.threshold_otsu(image_data)
        else:
            bin_wanted = (
                0 if assign_mid_to_fg else 1
            )
            threshold = skimage.filters.threshold_multiotsu(image_data, nbins=128)
            threshold = threshold[bin_wanted]
    else:
        raise ValueError("Invalid thresholding settings")
    return threshold

def get_local_threshold( image, mask, volumetric, adaptive_window_size):
    image_data = np.where(mask, image, np.nan)

    if len(image_data) == 0 or np.all(image_data == np.nan):
        local_threshold = np.zeros_like(image_data)

    elif np.all(image_data == image_data[0]):
        local_threshold = np.full_like(image_data, image_data[0])

    elif threshold_operation == TM_LI:
        local_threshold = self._run_local_threshold(
            image_data,
            method=skimage.filters.threshold_li,
            volumetric=volumetric,
            tolerance=max(np.min(np.diff(np.unique(image))) / 2, 0.5 / 65536)
        )
    elif threshold_operation == TM_OTSU:
        if two_class_otsu:
            local_threshold = _run_local_threshold(
                image_data,
                method=skimage.filters.threshold_otsu,
                volumetric=volumetric,
            )
        else:
            local_threshold = _run_local_threshold(
                image_data,
                method=skimage.filters.threshold_multiotsu,
                volumetric=volumetric,
                nbins=128,
            )

    elif threshold_operation == TM_SAUVOLA:
        image_data = np.where(mask, image, 0)
        adaptive_window = adaptive_window_size
        if adaptive_window % 2 == 0:
            adaptive_window += 1
        local_threshold = skimage.filters.threshold_sauvola(
            image_data, window_size=adaptive_window
        )

    else:
        raise ValueError("Invalid thresholding settings")
    return local_threshold

def _run_local_threshold(
                                image_data, 
                                method, 
                                volumetric=False, 
                                threshold_operation=TM_LI, 
                                automatic=False, 
                                two_class_otsu=False, 
                                assign_mid_to_fg = True,
                                adaptive_window_size=80,
                                **kwargs):
    if volumetric:
        t_local = np.zeros_like(image_data)
        for index, plane in enumerate(image_data):
            t_local[index] = _get_adaptive_threshold(plane, method, **kwargs)
    else:
        t_local = self._get_adaptive_threshold(image_data, method, **kwargs)
    return skimage.img_as_float(t_local)



def _get_adaptive_threshold(
                                mage_data, 
                                threshold_method, 
                                threshold_operation=TM_LI, 
                                automatic=False, 
                                two_class_otsu=False, 
                                assign_mid_to_fg = True,
                                adaptive_window_size=80,
                                **kwargs):
    """Given a global threshold, compute a threshold per pixel
    Break the image into blocks, computing the threshold per block.
    Afterwards, constrain the block threshold to .7 T < t < 1.5 T.
    """
    # for the X and Y direction, find the # of blocks, given the
    # size constraints
    if threshold_operation == TM_OTSU:
        bin_wanted = (
            0 if assign_middle_to_foreground.value == "Foreground" else 1
        )
    image_size = np.array(image_data.shape[:2], dtype=int)
    nblocks = image_size // self.adaptive_window_size.value
    if any(n < 2 for n in nblocks):
        raise ValueError(
            "Adaptive window cannot exceed 50%% of an image dimension.\n"
            "Window of %dpx is too large for a %sx%s image"
            % (adaptive_window_size, image_size[1], image_size[0])
        )
    #
    # Use a floating point block size to apportion the roundoff
    # roughly equally to each block
    #
    increment = np.array(image_size, dtype=float) / np.array(
        nblocks, dtype=float
    )
    #
    # Put the answer here
    #
    thresh_out = np.zeros(image_size, image_data.dtype)
    #
    # Loop once per block, computing the "global" threshold within the
    # block.
    #
    block_threshold = np.zeros([nblocks[0], nblocks[1]])
    for i in range(nblocks[0]):
        i0 = int(i * increment[0])
        i1 = int((i + 1) * increment[0])
        for j in range(nblocks[1]):
            j0 = int(j * increment[1])
            j1 = int((j + 1) * increment[1])
            block = image_data[i0:i1, j0:j1]
            block = block[~np.isnan(block)]
            if len(block) == 0:
                threshold_out = 0.0
            elif np.all(block == block[0]):
                # Don't compute blocks with only 1 value.
                threshold_out = block[0]
            elif (threshold_operation == TM_OTSU and
                    two_class_otsu and
                    len(np.unique(block)) < 3):
                # Can't run 3-class otsu on only 2 values.
                threshold_out = skimage.filters.threshold_otsu(block)
            else:
                try: 
                    threshold_out = threshold_method(block, **kwargs)
                except ValueError:
                    # Drop nbins kwarg when multi-otsu fails. See issue #6324 scikit-image
                    threshold_out = threshold_method(block)
            if isinstance(threshold_out, np.ndarray):
                # Select correct bin if running multiotsu
                threshold_out = threshold_out[bin_wanted]
            block_threshold[i, j] = threshold_out

    #
    # Use a cubic spline to blend the thresholds across the image to avoid image artifacts
    #
    spline_order = min(3, np.min(nblocks) - 1)
    xStart = int(increment[0] / 2)
    xEnd = int((nblocks[0] - 0.5) * increment[0])
    yStart = int(increment[1] / 2)
    yEnd = int((nblocks[1] - 0.5) * increment[1])
    xtStart = 0.5
    xtEnd = image_data.shape[0] - 0.5
    ytStart = 0.5
    ytEnd = image_data.shape[1] - 0.5
    block_x_coords = np.linspace(xStart, xEnd, nblocks[0])
    block_y_coords = np.linspace(yStart, yEnd, nblocks[1])
    adaptive_interpolation = scipy.interpolate.RectBivariateSpline(
        block_x_coords,
        block_y_coords,
        block_threshold,
        bbox=(xtStart, xtEnd, ytStart, ytEnd),
        kx=spline_order,
        ky=spline_order,
    )
    thresh_out_x_coords = np.linspace(
        0.5, int(nblocks[0] * increment[0]) - 0.5, thresh_out.shape[0]
    )
    thresh_out_y_coords = np.linspace(
        0.5, int(nblocks[1] * increment[1]) - 0.5, thresh_out.shape[1]
    )

    thresh_out = adaptive_interpolation(thresh_out_x_coords, thresh_out_y_coords)

    return thresh_out

def _correct_global_threshold(threshold, corr_value, threshold_range):
    threshold *= corr_value
    return min(max(threshold, threshold_range.min), threshold_range.max)

def _correct_local_threshold(t_local_orig, t_guide,threshold_correction_factor,threshold_range):
    t_local = t_local_orig.copy()

    # Constrain the local threshold to be within [0.7, 1.5] * global_threshold. It's for the pretty common case
    # where you have regions of the image with no cells whatsoever that are as large as whatever window you're
    # using. Without a lower bound, you start having crazy threshold s that detect noise blobs. And same for
    # very crowded areas where there is zero background in the window. You want the foreground to be all
    # detected.
    t_min = max(threshold_range.min, t_guide * 0.7)
    t_max = min(threshold_range.max, t_guide * 1.5)

    t_local[t_local < t_min] = t_min
    t_local[t_local > t_max] = t_max

    return t_local


def cp_adaptive_threshold(
                image_data,
                th_method, #skimage.filters.threshold_li,
                volumetric,
                window_size, 
                log_scale
                ):
    """   
    wrapper for the functions from CellProfiler
    NOTE: might work better to copy from CellProfiler/centrosome/threshold.py 
    https://github.com/CellProfiler/centrosome/blob/master/centrosome/threshold.py
    """

    th_guide = get_global_threshold(image_data, mask)
    th_original = get_local_threshold(image_data, mask, volumetric)


    final_threshold, orig_threshold, guide_threshold = get_threshold(
            input_image, 
            th_guide,
            th_original,
            log_scale=log_scale,
        )



    binary_image, _ = apply_threshold(input_image, final_threshold)
    return binary_image



def apply_threshold(image, threshold, mask=None, automatic=False):
    if mask is not None:
        return (data >= threshold) & mask
    else:
        return data>= threshold
    
        
def get_threshold(
            image, 
            th_guide,
            th_original,
            automatic=False, 
            log_scale=False
            ):

    need_transform = (
                threshold_operation in (TM_LI, TM_OTSU) and
                log_scale
        )

    if need_transform:
        image_data, conversion_dict = log_transform(image)
    else:
        image_data = image

    if  threshold_scope == TS_GLOBAL or automatic:
        th_guide = None
        th_original = get_global_threshold(image_data, image.mask, automatic=automatic)

    elif threshold_scope == TS_ADAPTIVE:
        th_guide = get_global_threshold(image_data, image.mask)
        th_original = get_local_threshold(image_data, image.mask, image.volumetric)
    else:
        raise ValueError("Invalid thresholding settings")

    if need_transform:
        th_original = inverse_log_transform(th_original, conversion_dict)
        if th_guide is not None:
            th_guide = inverse_log_transform(th_guide, conversion_dict)

    if threshold_scope == TS_GLOBAL or automatic:
        th_corrected = _correct_global_threshold(th_original)
    else:
        th_guide = _correct_global_threshold(th_guide)
        th_corrected = _correct_local_threshold(th_original, th_guide)

    return th_corrected, th_original, th_guide

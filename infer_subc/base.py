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
from skimage.morphology import remove_small_objects, ball, dilation, binary_closing
from skimage.filters import threshold_triangle, threshold_otsu, threshold_li
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from aicsimageio.writers import OmeTiffWriter
from napari_aicsimageio.core import  reader_function

import aicssegmentation
from aicssegmentation.core.seg_dot import dot_3d_wrapper, dot_slice_by_slice, dot_2d_slice_by_slice_wrapper, dot_3d
from aicssegmentation.core.pre_processing_utils import ( intensity_normalization, 
                                                         image_smoothing_gaussian_3d,  
                                                         image_smoothing_gaussian_slice_by_slice )
from aicssegmentation.core.utils import topology_preserving_thinning, hole_filling, size_filter
from aicssegmentation.core.MO_threshold import MO
from aicssegmentation.core.vessel import filament_2d_wrapper, vesselnessSliceBySlice
from aicssegmentation.core.output_utils import   save_segmentation,  generate_segmentation_contour
                                                 


# example constant variable
NAME = "infer_subc"


# ## we need to define some image processing wrappers... partials should work great
# from functools import partial


# these are the fuinctions that need to be set by each notebook...
# notebook workflow will produce the in_params dictionary nescessary 
# so all the images can be pushed through these functions (procedures)

##########################
# 1.  infer_NUCLEI
##########################
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

    med_filter_size = 4   
    gaussian_smoothing_truncate_range = 3.0
    struct_img = image_smoothing_gaussian_slice_by_slice(   struct_img,
                                                                                                        sigma=gaussian_smoothing_sigma,
                                                                                                        truncate_range = gaussian_smoothing_truncate_range
                                                                                                    )
    out_p["median_filter_size"] = med_filter_size 
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
    struct_obj = aicssegmentation.core.utils.hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    out_p['hole_max'] = hole_max

    small_object_max = 35
    struct_obj = aicssegmentation.core.utils.size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
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

    struct_obj = aicssegmentation.core.utils.hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    struct_obj = aicssegmentation.core.utils.size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
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
    struct_obj = aicssegmentation.core.utils.hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    out_p['hole_max'] = hole_max

    small_object_max = 35
    struct_obj = aicssegmentation.core.utils.size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
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
    struct_obj = aicssegmentation.core.utils.hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    struct_obj = aicssegmentation.core.utils.size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
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
    struct_obj = aicssegmentation.core.utils.hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    out_p['hole_max'] = hole_max

    small_object_max = 30
    struct_obj = aicssegmentation.core.utils.size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
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
    struct_obj = aicssegmentation.core.utils.hole_filling(struct_obj, hole_min =0. , hole_max=hole_max**2, fill_2d = True) 
    struct_obj = aicssegmentation.core.utils.size_filter(struct_obj, # wrapper to remove_small_objects which can do slice by slice
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


def infer_LYSOSOMES(struct_img, out_path, cyto_labels, in_params):
    pass

def infer_MITOCHONDRIA(struct_img, out_path, cyto_labels, in_params):
    pass

def infer_GOLGI(struct_img, out_path, cyto_labels, in_params):
    pass

def infer_PEROXISOMES(struct_img, out_path, cyto_labels, in_params):
    pass

def infer_ENDOPLASMIC_RETICULUM(struct_img, out_path, cyto_labels, in_params):
    pass

def infer_LIPID_DROPLET(struct_img, out_path, cyto_labels, in_params):
    pass


#### Median FIltering for 2D
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





def cp_adaptive_threshold(
                image_data,
                th_method, #skimage.filters.threshold_li,
                volumetric,
                window_size, 
                tolerance
                ):
    """   
    wrapper for the functions from CellProfiler
    NOTE: might work better to copy from CellProfiler/centrosome/threshold.py 
    https://github.com/CellProfiler/centrosome/blob/master/centrosome/threshold.py
    
    """
    
    threshold = _run_local_threshold(image_data, th_method, volumetric, window_size, tolerance) #**kwargs):
    return threshold



def _run_local_threshold(image_data, method, volumetric, window_size, tolerance): #**kwargs):
    if volumetric:
        t_local = np.zeros_like(image_data)
        for index, plane in enumerate(image_data):
            t_local[index] = _get_adaptive_threshold(plane, method, window_size, tolerance) #**kwargs)
    else:
        t_local = _get_adaptive_threshold(image_data, method, window_size, tolerance) #, **kwargs)
    return img_as_float(t_local)

def _get_adaptive_threshold(image_data, threshold_method, adaptive_window_size, tolerance):
    """Given a global threshold, compute a threshold per pixel
    Break the image into blocks, computing the threshold per block.
    Afterwards, constrain the block threshold to .7 T < t < 1.5 T.
    """
    # for the X and Y direction, find the # of blocks, given the
    # size constraints

    # if self.threshold_operation == TM_OTSU:
    #     bin_wanted = (
    #         0 if self.assign_middle_to_foreground.value == "Foreground" else 1
    #     )
    image_size = np.array(image_data.shape[:2], dtype=int)
    nblocks = image_size // adaptive_window_size
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
            # elif (self.threshold_operation == TM_OTSU and
            #       self.two_class_otsu.value == O_THREE_CLASS and
            #       len(np.unique(block)) < 3):
            #     # Can't run 3-class otsu on only 2 values.
            #     threshold_out = skimage.filters.threshold_otsu(block)
            else:
                try: 
                    threshold_out = threshold_method(block) #, tolerance=tolerance) #**kwargs)
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
    adaptive_interpolation = RectBivariateSpline(
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
            d[t.tag] = texts
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







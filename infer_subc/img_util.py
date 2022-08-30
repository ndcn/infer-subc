import numpy as np
from skimage.filters import (
    threshold_triangle, 
    threshold_otsu, 
    threshold_li,
    threshold_multiotsu)   

from scipy.ndimage import median_filter, extrema

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



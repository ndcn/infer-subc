from typing import Tuple, List, Union, Any
import numpy as np
from skimage.filters import threshold_triangle, threshold_otsu, threshold_li, threshold_multiotsu, threshold_sauvola
from skimage.morphology import white_tophat, ball, disk, black_tophat, label
from skimage.segmentation import clear_border, watershed

from scipy.ndimage import median_filter, extrema, distance_transform_edt, sum, minimum_filter, maximum_filter

from aicssegmentation.core.utils import size_filter, hole_filling
from aicssegmentation.core.vessel import vesselness2D
from aicssegmentation.core.MO_threshold import MO
from aicssegmentation.core.vessel import filament_2d_wrapper, filament_3d_wrapper
from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper, dot_3d_wrapper


def stack_layers(*layers) -> np.ndarray:
    """wrapper to stack the inferred objects into a single numpy.ndimage"""

    return np.stack(layers, axis=0)

### USED ###
def stack_masks(nuc_mask: np.ndarray, cellmask: np.ndarray, cyto_mask: np.ndarray) -> np.ndarray:
    """stack canonical masks:  cellmask, nuc, cytoplasm as uint8 (never more than 255 nuclei)"""
    layers = [nuc_mask, cellmask, cyto_mask]
    return np.stack(layers, axis=0).astype(np.uint8)


# TODO: check that the "noise" for the floor is correct... inverse_log should remove it?
### USED ###
def log_transform(image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Renormalize image intensities to log space

    Parameters
    ------------
    image:
        a 3d image

    Returns
    -------------
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

### USED ###
def inverse_log_transform(image: np.ndarray, d: dict) -> np.ndarray:
    """Convert the values in image back to the scale prior to log_transform

    Parameters
    ------------
    image:
        a 3d image
    d:
        dictionary returned by log_transform

    Returns
    -------------
        de-logged image (np.ndarray)
    """
    return np.exp(unstretch(image, d["log_min"], d["log_max"]))

### USED ###
def stretch(image: np.ndarray, mask: Union[np.ndarray, None] = None) -> np.ndarray:
    """Normalize an image to make the minimum zero and maximum one

    Parameters
    ------------
    image:
        a 3d image to be normalized
    mask:
        optional mask of relevant pixels. None (default) means don't mask

    Returns
    -------------
        stretched (normalized to [0,1]) image (np.ndarray)

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


def unstretch(image: np.ndarray, minval: Union[int, float], maxval: Union[int, float]) -> np.ndarray:
    """Perform the inverse of stretch, given a stretched image

    Parameters
    ------------
    image:
        an image stretched by stretch or similarly scaled value or values
    minval:
        minimum of previously stretched image
    maxval:
        maximum of previously stretched image

    Returns
    -------------
        stretched (normalized to [0,1]) image (np.ndarray)
    """
    return image * (maxval - minval) + minval

### USED ###
def threshold_li_log(image_in: np.ndarray) -> np.ndarray:
    """
    thin wrapper to log-scale and inverse log image for threshold finding using li minimum cross-entropy

    Parameters
    ------------
    image:
        an np.ndarray
    Returns
    -------------
        boolean np.ndarray

    """
    image, d = log_transform(image_in.copy())
    threshold = threshold_li(image)
    threshold = inverse_log_transform(threshold, d)
    return threshold

    # image_out = log_transform(image_in.copy())
    # li_thresholded = structure_img_smooth > threshold_li_log(structure_img_smooth)


def threshold_otsu_log(image_in):
    """
    thin wrapper to log-scale and inverse log image for threshold finding using otsu

    Parameters
    ------------
    image:
        an np.ndarray
    Returns
    -------------
        boolean np.ndarray
    """
    image, d = log_transform(image_in.copy())
    threshold = threshold_otsu(image)
    threshold = inverse_log_transform(threshold, d)
    return threshold


def threshold_multiotsu_log(image_in):
    """
    thin wrapper to log-scale and inverse log image for threshold finding using otsu

    Parameters
    ------------
    image:
        an np.ndarray
    Returns
    -------------
        boolean np.ndarray
    """
    image, d = log_transform(image_in.copy())
    thresholds = threshold_multiotsu(image)
    thresholds = inverse_log_transform(thresholds, d)
    return thresholds

### USED ###
def masked_object_thresh(
    structure_img_smooth: np.ndarray, global_method: str, cutoff_size: int, local_adjust: float
) -> np.ndarray:
    """
    wrapper for applying Masked Object Thresholding with just two parameters via `MO` from `aicssegmentation`

    Parameters
    ------------
    structure_img_smooth:
        a 3d image
    global_method:
         which method to use for calculating global threshold. Options include:
         "triangle", "median", and "ave_tri_med".
         "ave_tri_med" refers the average of "triangle" threshold and "mean" threshold.
    cutoff_size:
        Masked Object threshold `size_min`
    local_adjust:
        Masked Object threshold `local_adjust`

    Returns
    -------------
        np.ndimage

    """
    struct_obj = MO(
        structure_img_smooth,
        object_minArea=cutoff_size,
        global_thresh_method=global_method,
        extra_criteria=True,
        local_adjust=local_adjust,
        return_object=False,
        dilate=False,  # WARNING: dilate=True causes a bug if there is only one Z
    )
    return struct_obj


def get_interior_labels(img_in: np.ndarray) -> np.ndarray:
    """
    gets the labeled objects from the X,Y "interior" of the image. We only want to clear the objects touching the sides of the volume, but not the top and bottom, so we pad and crop the volume along the 0th axis

    Parameters
    ------------
    img_in:
        a 3d image

    Returns
    -------------
        np.ndimage of labeled segmentations NOT touching the sides

    """
    segmented_padded = np.pad(
        label(img_in),
        ((1, 1), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    interior = clear_border(segmented_padded)[1:-1]
    return interior
    # relabel?
    # return label(interior).astype(np.uint16)

### USED ###
def label_uint16(in_obj: np.ndarray) -> np.ndarray:
    """
    label segmentation and return as uint16

    Parameters
    ------------
    in_obj:
        a 3d image segmentaiton

    Returns
    -------------
        np.ndimage of labeled segmentations as np.uint16

    """
    if in_obj.dtype == "bool":
        return label(in_obj).astype(np.uint16)
    else:  # in_obj.dtype == np.uint8:
        return label(in_obj > 0).astype(np.uint16)

### USED ###
def label_bool_as_uint16(in_obj: np.ndarray) -> np.ndarray:
    """
    label segmentation and return as uint16

    Parameters
    ------------
    in_obj:
        a 3d image segmentaiton

    Returns
    -------------
        np.ndimage of labeled segmentations as np.uint16

    """
    return (in_obj > 0).astype(np.uint16)


def median_filter_slice_by_slice(struct_img: np.ndarray, size: int) -> np.ndarray:
    """
    wrapper for applying 2D median filter slice by slice on a 3D image

    Parameters
    ------------
    img:
        a 3d image

    size:
        the linear "size" which will be squared for

    Returns
    -------------
        np.ndimage

    """
    structure_img_denoise = np.zeros_like(struct_img)

    # this might be faster:  scipy.signal.medfilt2d()
    for zz in range(struct_img.shape[0]):
        structure_img_denoise[zz, :, :] = median_filter(struct_img[zz, :, :], size=size)

    return structure_img_denoise

### USED ###
def min_max_intensity_normalization(struct_img: np.ndarray) -> np.ndarray:
    """Normalize the intensity of input image so that the value range is from 0 to 1.

    Parameters
    ------------
    img:
        a 3d image

    Returns
    -------------
        np.ndimage
    """
    strech_min = struct_img.min()
    strech_max = struct_img.max()
    # do we need to convert to float?
    # #.astype(np.double)
    struct_img = (struct_img - strech_min + 1e-8) / (strech_max - strech_min + 1e-8)

    return struct_img


# # depriated. using media/gauss filtering for ER now
# def normalized_edge_preserving_smoothing(img_in: np.ndarray) -> np.ndarray:
#     """wrapper to min-max normalize + aicssegmentaion. edge_preserving_smoothing_3d

#     Parameters
#     ------------
#     img:
#         a 3d image

#     Returns
#     -------------
#         smoothed and normalized np.ndimage
#     """
#     struct_img = min_max_intensity_normalization(img_in)
#     # edge-preserving smoothing (Option 2, used for Sec61B)
#     struct_img = edge_preserving_smoothing_3d(struct_img)
#     # 10 seconds!!!!  tooooo slow... for maybe no good reason
#     return min_max_intensity_normalization(struct_img)

### USED ###
def weighted_aggregate(img_in: np.ndarray, *weights: int) -> np.ndarray:
    """
    helper to find weighted sum images
    Parameters
    ------------
    img_in:
        a 3d imagenp.ndarray
    *weights:
        list of integer weights to apply to our channels.  if the weights are less than 1, they will NOT be included (and might not be there)

    Returns
    -------------
        np.ndimage of weighted sum of channels

    """

    img_out = np.zeros_like(img_in[0]).astype(np.double)
    for ch, w in enumerate(weights):
        if w > 0:
            img_out += (w * 1.0) * img_in[ch]

    return img_out

### USED ###
def make_aggregate(
    img_in: np.ndarray,
    weight_ch0: int = 0,
    weight_ch1: int = 0,
    weight_ch2: int = 0,
    weight_ch3: int = 0,
    weight_ch4: int = 0,
    weight_ch5: int = 0,
    weight_ch6: int = 0,
    weight_ch7: int = 0,
    weight_ch8: int = 0,
    weight_ch9: int = 0,
    rescale: bool = True,
) -> np.ndarray:
    """define multi_channel aggregate.  weighted sum wrapper (plugin)

    Parameters
    ------------
    w0,w1,w2,w3,w4,w5,w6,w7,w8,w9
        channel weights
    rescale:
        scale to [0,1] if True. default True

    Returns
    -------------
        np.ndarray scaled aggregate

    """
    weights = (weight_ch0, weight_ch1, weight_ch2, weight_ch3, weight_ch4,
               weight_ch5, weight_ch6, weight_ch7, weight_ch8, weight_ch9)
    if rescale:
        # TODO: might NOT overflow here... maybe NOT do the normaization first?
        return min_max_intensity_normalization(weighted_aggregate(min_max_intensity_normalization(img_in), *weights))
    else:
        return weighted_aggregate(img_in, *weights)

### USED ###
def apply_threshold(
    img_in: np.ndarray,
    method: str = "otsu",
    thresh_factor: float = 1.0,
    thresh_min: Union[None, float] = None,
    thresh_max: Union[None, float] = None,
) -> np.ndarray:
    """return a binary mask after applying a log_li threshold

    Parameters
    ------------
    img_in:
        np.ndarray input image
    method:
        method for applying threshold.  "otsu"  or "li" (default), "triangle", "median", "ave", "sauvola","multi_otsu","muiltiotsu"
    thresh_factor:
        scaling value for threshold, defaults=1.0
    thresh_min
        absolute minumum for threshold, default=None
    thresh_max
        absolute maximum for threshold, default=None

    Returns
    -------------
        thresholded boolean np.ndarray
    """

    if method == "tri" or method == "triangle":
        threshold_val = threshold_triangle(img_in)
    elif method == "med" or method == "median":
        threshold_val = np.percentile(img_in, 50)
    elif method == "ave" or method == "ave_tri_med":
        global_tri = threshold_triangle(img_in)
        global_median = np.percentile(img_in, 50)
        threshold_val = (global_tri + global_median) / 2
    elif method == "li" or method == "cross_entropy" or method == "crossentropy":
        threshold_val = threshold_li(img_in)
    elif method == "sauvola":
        threshold_val = threshold_sauvola(img_in)
    elif method == "mult_otsu" or method == "multiotsu":
        threshold_val = threshold_multiotsu(img_in)
    else:  # default to "otsu"
        threshold_val = threshold_otsu(img_in)

    threshold = threshold_val * thresh_factor

    if thresh_min is not None:
        threshold = max(threshold, thresh_min)
    if thresh_max is not None:
        threshold = min(threshold, thresh_max)
    return img_in > threshold

### USED ###
def apply_log_li_threshold(
    img_in: np.ndarray,
    thresh_factor: float = 1.0,
    thresh_min: Union[None, float] = None,
    thresh_max: Union[None, float] = None,
) -> np.ndarray:
    """return a binary mask after applying a log_li threshold

    Parameters
    ------------
    img_in:
        input ndimage array (np.ndimage)
    thresh_factor:
        scaling value for threshold, defaults=1.0
    thresh_min
        absolute minumum for threshold, default=None
    thresh_max
        absolute maximum for threshold, default=None

    Returns
    -------------
        thresholded boolean np.ndarray
    """
    # struct_obj = struct_img > filters.threshold_li(struct_img)
    threshold_value_log = threshold_li_log(img_in)
    threshold = threshold_value_log * thresh_factor

    if thresh_min is not None:
        threshold = max(threshold, thresh_min)
    if thresh_max is not None:
        threshold = min(threshold, thresh_max)
    return img_in > threshold


# NOTE this is identical to veselnessSliceBySlice from aicssegmentation.core.vessel
def vesselness_slice_by_slice(nd_array: np.ndarray, sigma: float, cutoff: float = -1, tau: float = 0.75):
    """
    wrapper for applying multi-scale 2D filament filter on 3D images in a
    slice by slice fashion,  Note that it only performs at a single scale....     NOTE: The paramater
    whiteonblack = True is hardcoded which sets the filamentous structures are bright on dark background

    Parameters
    -----------
    nd_array:
        the 3D image to be filterd on
    sigma:
        single scale to use
    cutoff:
        the cutoff value to apply on the filter result. If the cutoff is
        negative, no cutoff will be applied. Default is -1.
    tau:
        parameter that controls response uniformity. The value has to be
        between 0.5 and 1. Lower tau means more intense output response.
        Default is 0.5
    """

    # # this hack is to accomodate the workflow widgets
    # if not isinstance(sigmas, List):
    #     sigmas = [sigmas]

    mip = np.amax(nd_array, axis=0)
    response = np.zeros(nd_array.shape)
    for zz in range(nd_array.shape[0]):
        tmp = np.concatenate((nd_array[zz, :, :], mip), axis=1)
        tmp = vesselness2D(tmp, sigmas=[sigma], tau=tau, whiteonblack=True)
        response[zz, :, : nd_array.shape[2] - 3] = tmp[:, : nd_array.shape[2] - 3]

    if cutoff < 0:
        return response
    else:
        return response > cutoff

### USED ###
def select_channel_from_raw(img_in: np.ndarray, chan: Union[int, Tuple[int]]) -> np.ndarray:
    """ "
    select channel from multi-channel 3D image (np.ndarray)
    Parameters
    ------------
    img_in :
        the 3D image to be filterd on
    chan :
        channel to extract.

    Returns
    -------------
        np.ndarray
    """
    return img_in[chan]


def select_z_from_raw(img_in: np.ndarray, z_slice: Union[int, Tuple[int]]) -> np.ndarray:
    """
    select Z-slice from 3D multi-channel image (np.ndarray)

    Parameters
    ------------
    img_in :
        the 3D image to be filterd on
    chan :
        channel to extract.

    Returns
    -------------
        np.ndarray
    """
    if isinstance(z_slice, int):
        z_slice = [z_slice]
    else:
        z_slice = list(z_slice)

    return img_in[:, z_slice, :, :]

### USED ###
def scale_and_smooth(
    img_in: np.ndarray, median_size: int = 1, gauss_sigma: float = 1.34, slice_by_slice: bool = True
) -> np.ndarray:
    """
    helper to perform min-max scaling, and median+gaussian smoothign all at once
    Parameters
    ------------
    img_in: np.ndarray
        a 3d image
    median_size: int
        width of median filter for signal
    gauss_sigma: float
        sigma for gaussian smoothing of  signal
    slice_by_slice:
        NOT IMPLIMENTED.  toggles whether to do 3D operations or slice by slice in Z

    Returns
    -------------
        np.ndimage

    """
    img = min_max_intensity_normalization(img_in.copy())  # is this copy nescesa

    # TODO:  make non-slice-by-slice work
    slice_by_slice = True
    if slice_by_slice:
        if median_size > 1:
            img = median_filter_slice_by_slice(img, size=median_size)
        img = image_smoothing_gaussian_slice_by_slice(img, sigma=gauss_sigma)
    else:
        print(" PLEASE CHOOOSE 'slice-by-slice', 3D is not yet implimented")

    return img


# DEPRICATED
def aggregate_signal_channels(
    img_in: np.ndarray, chs: Union[List, Tuple], ws: Union[List, Tuple, Any] = None
) -> np.ndarray:
    """
    return a weighted sum of the image across channels (DEPRICATED)

    Parameters
    ------------
    img_in:
        np.ndarray  [ch,z,x,y]
    chs:
        list/tuple of channels to aggregate
    ws:
        list/tuple/ of weights for aggregation

    Returns
    -------------
        np.ndarray
    """
    n_chan = len(chs)
    if n_chan <= 1:
        return img_in[chs]

    if ws is None:
        ws = n_chan * [1.0]
    img_out = np.zeros_like(img_in[0]).astype(np.double)
    for w, ch in zip(ws, chs):
        img_out += w * img_in[ch]
    return img_out
    # return img_in[ch_to_agg,].astype( np.double ).sum(axis=0)


def choose_agg_signal_zmax(img_in: np.ndarray, chs: List[int], ws=None, mask=None) -> np.ndarray:
    """
    return z the maximum signal for the aggregate signal

    Parameters
    ------------
    img_in:
        np.ndarray  [ch,z,x,y]
    chs:
        list of channels to aggregate
    ws:
        list of weights for aggregation
    mask:
        mask for img_in

    Returns
    -------------
        np.ndarray z with maximum signal
    """
    total_florescence_ = aggregate_signal_channels(img_in, chs)
    if mask is not None:
        total_florescence_[mask] = 0.0
    return int(total_florescence_.sum(axis=(1, 2)).argmax())

### USED ###
def masked_inverted_watershed(img_in: np.ndarray, 
                                     markers: np.ndarray, 
                                     mask: np.ndarray, 
                                     method: str='slice-by-slice'):
    """wrapper for watershed on inverted image and masked

    Parameters
    ------------
    in_img:
        a 3d image
    markers: 
        objects used to seed the watershed
    mask:
        instance segmentation of the in_img
    method:
        'slice-by-slice' results in a connectivity of np.ones((1,3,3), bool); '3D' results in a connectivity of np.ones((3,3,3), bool)

    """
    if method == 'slice-by-slice':    
        labels_out = watershed(
            1.0 - img_in,
            markers=markers,
            connectivity=np.ones((1, 3, 3), bool),
            mask=mask)
    elif method == '3D':
        labels_out = watershed(
        1.0 - img_in,
        markers=markers,
        connectivity=np.ones((3, 3, 3), bool),
        mask=mask)
    else:
        print(f"incompatable method: {method}")

    return labels_out



def choose_max_label(
    raw_signal: np.ndarray, labels_in: np.ndarray, target_labels: Union[np.ndarray, None] = None
) -> np.ndarray:
    """
    keep only the segmentation corresponding to the maximum raw signal.  candidate  label is taken from target_labels if not None

    Parameters
    ------------
    raw_signal:
        the image to filter on
    labels_in:
        segmentation labels
    target_labels:
        labels to consider

    Returns
    -------------
        np.ndarray of labels corresponding to the largest total signal

    """
    keep_label = get_max_label(raw_signal, labels_in, target_labels)
    labels_max = np.zeros_like(labels_in)
    labels_max[labels_in == keep_label] = 1
    return labels_max

### USED ###
def get_max_label(
    raw_signal: np.ndarray, labels_in: np.ndarray, target_labels: Union[np.ndarray, None] = None
) -> np.ndarray:
    """
    keep only the label with the maximum raw signal.  candidate  label is taken from target_labels if not None

    Parameters
    ------------
    raw_signal:
        the image to filter on
    labels_in:
        segmentation labels
    target_labels:
        labels to consider

    Returns
    -------------
        np.ndarray of labels corresponding to the largest total signal

    """
    if target_labels is None:
        all_labels = np.unique(labels_in)[1:]
    else:
        all_labels = np.unique(target_labels)[1:]

    total_signal = [raw_signal[labels_in == label].sum() for label in all_labels]
    # combine NU and "labels" to make a CELLMASK
    keep_label = all_labels[np.argmax(total_signal)]

    return keep_label

### USED ###
def fill_and_filter_linear_size(
    img: np.ndarray, hole_min: int, hole_max: int, min_size: int, method: str = "slice_by_slice", connectivity: int = 1
) -> np.ndarray:
    """wraper to aiscsegmentation `hole_filling` and `size_filter` with size argument in linear units

    Parameters
    ------------
    img:
        the image to filter on
    hole_min: int
        the minimum width of the holes to be filled
    hole_max: int
        the maximum width of the holes to be filled
    min_size: int
        the minimum size expressed as 1D length (so squared for slice-by-slice, cubed for 3D)
    method: str
        either "3D" or "slice_by_slice", default is "slice_by_slice"
    connnectivity: int
        the connectivity to use when computing object size
    Returns
    -------------
        a binary image after hole filling and filtering small objects; np.ndarray
    """
    if not img.any():
        return img

    if method == "3D":
        if hole_max > 0:
            img = hole_filling(img, hole_min=hole_min**3, hole_max=hole_max**3, fill_2d=False)
        return size_filter(img, min_size=min_size**3, method="3D", connectivity=connectivity)
    elif method == "slice_by_slice":
        if hole_max > 0:
            img = hole_filling(img, hole_min=hole_min**2, hole_max=hole_max**2, fill_2d=True)
        return size_filter(img, min_size=min_size**2, method="slice_by_slice", connectivity=connectivity)
    else:
        print(f"undefined method: {method}")


def size_filter_linear_size(
    img: np.ndarray, min_size: int, method: str = "slice_by_slice", connectivity: int = 1
) -> np.ndarray:
    """size filter wraper to aiscsegmentation `size_filter` with size argument in linear units

    Parameters
    ------------
    img:
        the image to filter on
    min_size: int
        the minimum size expressed as 1D length (so squared for slice-by-slice, cubed for 3D)
    method: str
        either "3D" or "slice_by_slice", default is "slice_by_slice"
    connnectivity: int
        the connectivity to use when computing object size
    Returns
    -------------
        np.ndarray
    """
    # return remove_small_objects(img > 0, min_size=min_size, connectivity=connectivity, in_place=False)
    if not img.any():
        return img

    if method == "3D":
        return size_filter(img, min_size=min_size**3, method="3D", connectivity=connectivity)
    elif method == "slice_by_slice":
        return size_filter(img, min_size=min_size**2, method="slice_by_slice", connectivity=connectivity)
    else:
        raise NotImplementedError(f"unsupported method {method}")


def hole_filling_linear_size(img: np.ndarray, hole_min: int, hole_max: int, fill_2d=True) -> np.ndarray:
    """Fill holes  wraper to aiscsegmentation `hole_filling` with size argument in linear units.  always does slice-by-slice

    Parameters
    ------------
    img:
        the image to filter on
    hole_min: int
        the minimum width of the holes to be filled
    hole_max: int
        the maximum width of the holes to be filled

    Returns
    -----------
        a binary image after hole filling
    """
    if fill_2d:
        return hole_filling(img, hole_min=hole_min**2, hole_max=hole_max**2, fill_2d=True)
    else:
        return hole_filling(img, hole_min=hole_min**3, hole_max=hole_max**3, fill_2d=False)

### USED ###
def apply_mask(img_in: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask the image

    Parameters
    ------------
    img_in:
        the image to filter on
    mask:
        the mask to apply

    Returns
    -----------
    img_out:
        a new (copied) array with mask applied
    """
    assert img_in.shape == mask.shape

    img_out = img_in.copy()
    if mask.dtype == "bool":
        img_out[~mask] = 0
    else:
        img_out[mask < 1] = 0

    return img_out


def enhance_speckles(image: np.ndarray, radius: int, volumetric: bool = False) -> np.ndarray:
    """enhance "spreckles" small dots

    Parameters
    ------------
    image: np.ndarray
        the image to filter on
    radius: int
        radius of the "filter"
    volumetric: bool
        True for 3D analysis

    Returns
    -----------
    result:
        filtered boolean np.ndarray
    """
    radius = radius / 2
    if volumetric:
        selem = ball(radius)
    else:
        selem = disk(radius)

    # if radius >10:
    #         minimum = scipy.ndimage.minimum_filter(image, footprint=selem)
    #         maximum = scipy.ndimage.maximum_filter(minimum, footprint=selem)
    #         result = data - maximum
    # else:
    result = white_tophat(image)

    return result


def enhance_neurites(image: np.ndarray, radius: int, volumetric: bool = False) -> np.ndarray:
    """enhance "neurites" or filiments

    Parameters
     ------------
     image: np.ndarray
         the image to filter on
     radius: int
         radius of the "filter"
     volumetric: bool
         True for 3D analysis

     Returns
     -----------
     result:
         filtered boolean np.ndarray
    """
    if volumetric:
        selem = ball(radius)
    else:
        selem = disk(radius)
    white = white_tophat(image, selem)
    black = black_tophat(image, selem)
    result = image + white - black
    result[result > 1] = 1
    result[result < 0] = 0

    return result

### USED ###
def filament_filter_3(in_img: np.ndarray, 
                       filament_scale_1: float, 
                       filament_cutoff_1: float,
                       filament_scale_2: float, 
                       filament_cutoff_2: float,
                       filament_scale_3: float, 
                       filament_cutoff_3: float,
                       method: str
                       ) -> np.ndarray:
    """filament filter helper function for 3 levels (scale+cut). filter pairs are run if scale is > 0.

    Parameters
    ------------
    in_img:
        the image to filter on np.ndarray
    filament_scale_1:
        scale or size of the "filter" float
    filament_cutoff_1:
        cutoff for thresholding float
    filament_scale_2:
        scale or size of the "filter" float
    filament_cutoff_2:
        cutoff for thresholding float
    filament_scale_3:
        scale or size of the "filter" float
    filament_cutoff_3:
        cutoff for thresholding float
    method:
        either "3D" or "slice_by_slice", default is "slice_by_slice"

    Returns
    -----------
    result:
        filtered boolean np.ndarray

    """
    scales = [filament_scale_1, filament_scale_2, filament_scale_3]
    cuts = [filament_cutoff_1, filament_cutoff_2, filament_cutoff_3]
    f_param = [[sc, ct] for sc, ct in zip(scales, cuts) if sc > 0]

    if method == "3D":
        seg = filament_3d_wrapper(in_img, f_param)
    elif method == "slice_by_slice":
        seg = filament_2d_wrapper(in_img, f_param)
    else:
        print(f"undefined method: {method}")

    return seg


def filament_filter(in_img: np.ndarray, filament_scale: float, filament_cut: float) -> np.ndarray:
    """filament wrapper to properly pack parameters into filament_2d_wrapper

    Parameters
    ------------
    in_img:
        the image to filter on np.ndarray
    filament_scale:
        scale or size of the "filter" float
    filament_cut:
        cutoff for thresholding float

    Returns
    -----------
    result:
        filtered boolean np.ndarray

    """
    f2_param = [[filament_scale, filament_cut]]
    # f2_param = [[1, 0.15]]  # [scale_1, cutoff_1]
    return filament_2d_wrapper(in_img, f2_param)


# def dot_filter_3(
#     in_img: np.ndarray,
#     dot_scale_1: float,
#     dot_cut_1: float,
#     dot_scale_2: float,
#     dot_cut_2: float,
#     dot_scale_3: float,
#     dot_cut_3: float,
# ) -> np.ndarray:
#     """spot filter helper function for 3 levels (scale+cut).  if scale_i is > 0.0001 its skipped

#     Parameters
#     ------------
#     in_img:
#         a 3d  np.ndarray image of the inferred organelle (labels or boolean)
#     dot_scale_1:
#         scale or size of the "filter" float
#     dot_cut_1:
#         cutoff for thresholding float
#     dot_scale_2:
#         scale or size of the "filter" float
#     dot_cut_2:
#         cutoff for thresholding float
#     dot_scale_3:
#         scale or size of the "filter" float
#     dot_cut_3:
#         cutoff for thresholding float

#     Returns
#     -------------
#     segmented dots over 3 scales

#     """
#     scales = [dot_scale_1, dot_scale_2, dot_scale_3]
#     cuts = [dot_cut_1, dot_cut_2, dot_cut_3]

#     s2_param = [[sc, ct] for sc, ct in zip(scales, cuts) if sc > 0.0001]
#     # s2_param = [[dot_scale1, dot_cut1], [dot_scale2, dot_cut2], [dot_scale3, dot_cut3]]
#     return dot_2d_slice_by_slice_wrapper(in_img, s2_param)

### USED ###
def dot_filter_3(
    in_img: np.ndarray,
    dot_scale_1: float,
    dot_cutoff_1: float,
    dot_scale_2: float,
    dot_cutoff_2: float,
    dot_scale_3: float,
    dot_cutoff_3: float,
    method: str = "slice_by_slice"
) -> np.ndarray:
    """spot filter helper function for 3 levels (scale+cut). filter pairs are run if scale is > 0.

    Parameters
    ------------
    in_img:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    dot_scale_1:
        scale or size of the "filter" float
    dot_cutoff_1:
        cutoff for thresholding float
    dot_scale_2:
        scale or size of the "filter" float
    dot_cutoff_2:
        cutoff for thresholding float
    dot_scale_3:
        scale or size of the "filter" float
    dot_cut_3:
        cutoff for thresholding float
    method:
        either "3D" or "slice_by_slice", default is "slice_by_slice"

    Returns
    -------------
    segmented dots over 3 scales

    """
    scales = [dot_scale_1, dot_scale_2, dot_scale_3]
    cuts = [dot_cutoff_1, dot_cutoff_2, dot_cutoff_3]
    s_param = [[sc, ct] for sc, ct in zip(scales, cuts) if sc > 0]

    if method == "3D":
        seg = dot_3d_wrapper(in_img, s_param)
    elif method == "slice_by_slice":
        seg = dot_2d_slice_by_slice_wrapper(in_img, s_param)
    else:
        print(f"undefined method: {method}")

    return seg

# centrosome routines


def size_similarly(labels, secondary):
    """Size the secondary matrix similarly to the labels matrix

    labels - labels matrix
    secondary - a secondary image or labels matrix which might be of
                different size.
    Return the resized secondary matrix and a mask indicating what portion
    of the secondary matrix is bogus (manufactured values).

    Either the mask is all ones or the result is a copy, so you can
    modify the output within the unmasked region w/o destroying the original.
    """
    if labels.shape[:2] == secondary.shape[:2]:
        return secondary, np.ones(secondary.shape, bool)
    if labels.shape[0] <= secondary.shape[0] and labels.shape[1] <= secondary.shape[1]:
        if secondary.ndim == 2:
            return (
                secondary[: labels.shape[0], : labels.shape[1]],
                np.ones(labels.shape, bool),
            )
        else:
            return (
                secondary[: labels.shape[0], : labels.shape[1], :],
                np.ones(labels.shape, bool),
            )

    #
    # Some portion of the secondary matrix does not cover the labels
    #
    result = np.zeros(list(labels.shape) + list(secondary.shape[2:]), secondary.dtype)
    i_max = min(secondary.shape[0], labels.shape[0])
    j_max = min(secondary.shape[1], labels.shape[1])
    if secondary.ndim == 2:
        result[:i_max, :j_max] = secondary[:i_max, :j_max]
    else:
        result[:i_max, :j_max, :] = secondary[:i_max, :j_max, :]
    mask = np.zeros(labels.shape, bool)
    mask[:i_max, :j_max] = 1
    return result, mask


def distance_to_edge(labels):
    """Compute the distance of a pixel to the edge of its object

    labels - a labels matrix

    returns a matrix of distances
    """
    colors = color_labels(labels)
    max_color = np.max(colors)
    result = np.zeros(labels.shape)
    if max_color == 0:
        return result

    for i in range(1, max_color + 1):
        mask = colors == i
        result[mask] = distance_transform_edt(mask)[mask]
    return result


def color_labels(labels, distance_transform=False):
    """Color a labels matrix so that no adjacent labels have the same color

    distance_transform - if true, distance transform the labels to find out
         which objects are closest to each other.

    Create a label coloring matrix which assigns a color (1-n) to each pixel
    in the labels matrix such that all pixels similarly labeled are similarly
    colored and so that no similiarly colored, 8-connected pixels have
    different labels.

    You can use this function to partition the labels matrix into groups
    of objects that are not touching; you can then operate on masks
    and be assured that the pixels from one object won't interfere with
    pixels in another.

    returns the color matrix
    """
    if distance_transform:
        i, j = distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
        dt_labels = labels[i, j]
    else:
        dt_labels = labels
    # Get the neighbors for each object
    v_count, v_index, v_neighbor = find_neighbors(dt_labels)
    # Quickly get rid of labels with no neighbors. Greedily assign
    # all of these a color of 1
    v_color = np.zeros(len(v_count) + 1, int)  # the color per object - zero is uncolored
    zero_count = v_count == 0
    if np.all(zero_count):
        # can assign all objects the same color
        return (labels != 0).astype(int)
    v_color[1:][zero_count] = 1
    v_count = v_count[~zero_count]
    v_index = v_index[~zero_count]
    v_label = np.argwhere(~zero_count).transpose()[0] + 1
    # If you process the most connected labels first and use a greedy
    # algorithm to preferentially assign a label to an existing color,
    # you'll get a coloring that uses 1+max(connections) at most.
    #
    # Welsh, "An upper bound for the chromatic number of a graph and
    # its application to timetabling problems", The Computer Journal, 10(1)
    # p 85 (1967)
    #
    sort_order = np.lexsort([-v_count])
    v_count = v_count[sort_order]
    v_index = v_index[sort_order]
    v_label = v_label[sort_order]
    for i in range(len(v_count)):
        neighbors = v_neighbor[v_index[i] : v_index[i] + v_count[i]]
        colors = np.unique(v_color[neighbors])
        if colors[0] == 0:
            if len(colors) == 1:
                # only one color and it's zero. All neighbors are unlabeled
                v_color[v_label[i]] = 1
                continue
            else:
                colors = colors[1:]
        # The colors of neighbors will be ordered, so there are two cases:
        # * all colors up to X appear - colors == np.arange(1,len(colors)+1)
        # * some color is missing - the color after the first missing will
        #   be mislabeled: colors[i] != np.arange(1, len(colors)+1)
        crange = np.arange(1, len(colors) + 1)
        misses = crange[colors != crange]
        if len(misses):
            color = misses[0]
        else:
            color = len(colors) + 1
        v_color[v_label[i]] = color
    return v_color[labels]


def find_neighbors(labels):
    """Find the set of objects that touch each object in a labels matrix

    Construct a "list", per-object, of the objects 8-connected adjacent
    to that object.
    Returns three 1-d arrays:
    * array of #'s of neighbors per object
    * array of indexes per object to that object's list of neighbors
    * array holding the neighbors.

    For instance, say 1 touches 2 and 3 and nobody touches 4. The arrays are:
    [ 2, 1, 1, 0], [ 0, 2, 3, 4], [ 2, 3, 1, 1]
    """
    max_label = np.max(labels)
    # Make a labels matrix with zeros around the edges so we can do index
    # offsets without worrying.
    #
    new_labels = np.zeros(np.array(labels.shape) + 2, labels.dtype)
    new_labels[1:-1, 1:-1] = labels
    labels = new_labels
    # Only consider the points that are next to others
    adjacent_mask = adjacent(labels)
    adjacent_i, adjacent_j = np.argwhere(adjacent_mask).transpose()
    # Get matching vectors of labels and neighbor labels for the 8
    # compass directions.
    count = len(adjacent_i)
    if count == 0:
        return (np.zeros(max_label, int), np.zeros(max_label, int), np.zeros(0, int))
    # The following bizarre construct does the following:
    # labels[adjacent_i, adjacent_j] looks up the label for each pixel
    # [...]*8 creates a list of 8 references to it
    # np.hstack concatenates, giving 8 repeats of the list
    v_label = np.hstack([labels[adjacent_i, adjacent_j]] * 8)
    v_neighbor = np.zeros(count * 8, int)
    index = 0
    for i, j in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        v_neighbor[index : index + count] = labels[adjacent_i + i, adjacent_j + j]
        index += count
    #
    # sort by label and neighbor
    #
    sort_order = np.lexsort((v_neighbor, v_label))
    v_label = v_label[sort_order]
    v_neighbor = v_neighbor[sort_order]
    #
    # eliminate duplicates by comparing each element after the first one
    # to its previous
    #
    first_occurrence = np.ones(len(v_label), bool)
    first_occurrence[1:] = (v_label[1:] != v_label[:-1]) | (v_neighbor[1:] != v_neighbor[:-1])
    v_label = v_label[first_occurrence]
    v_neighbor = v_neighbor[first_occurrence]
    #
    # eliminate neighbor = self and neighbor = background
    #
    to_remove = (v_label == v_neighbor) | (v_neighbor == 0)
    v_label = v_label[~to_remove]
    v_neighbor = v_neighbor[~to_remove]
    #
    # The count of # of neighbors
    #
    v_count = fixup_scipy_ndimage_result(sum(np.ones(v_label.shape), v_label, np.arange(max_label, dtype=np.int32) + 1))
    v_count = v_count.astype(int)
    #
    # The index into v_neighbor
    #
    v_index = np.cumsum(v_count)
    v_index[1:] = v_index[:-1]
    v_index[0] = 0
    return (v_count, v_index, v_neighbor)


# from CellProfiler
def fixup_scipy_ndimage_result(whatever_it_returned):
    """Convert a result from scipy.ndimage to a numpy array

    scipy.ndimage has the annoying habit of returning a single, bare
    value instead of an array if the indexes passed in are of length 1.
    For instance:
    scipy.ndimage.maximum(image, labels, [1]) returns a float
    but
    scipy.ndimage.maximum(image, labels, [1,2]) returns a list
    """
    if getattr(whatever_it_returned, "__getitem__", False):
        return np.array(whatever_it_returned)
    else:
        return np.array([whatever_it_returned])


def adjacent(labels):
    """Return a binary mask of all pixels which are adjacent to a pixel of
    a different label.

    """
    high = labels.max() + 1
    if high > np.iinfo(labels.dtype).max:
        labels = labels.astype(np.int32)
    image_with_high_background = labels.copy()
    image_with_high_background[labels == 0] = high
    min_label = minimum_filter(
        image_with_high_background,
        footprint=np.ones((3, 3), bool),
        mode="constant",
        cval=high,
    )
    max_label = maximum_filter(labels, footprint=np.ones((3, 3), bool), mode="constant", cval=0)
    return (min_label != max_label) & (labels > 0)


def img_to_uint8(data_in: np.ndarray) -> np.ndarray:
    """
    helper to convert bask to `binary` uint8 (true -> 255) to accomodate napari default scaling
    """
    print(f"changing from {data_in.dtype} to np.uint8")
    data_in = data_in.astype(np.uint8)
    data_in[data_in > 0] = 1
    return data_in


def img_to_bool(data_in: np.ndarray) -> np.ndarray:
    """
    helper to make sure we are keeping track of things correctly
    """
    print(f"changing from {data_in.dtype} to bool")
    data_out = data_in > 0
    print(f"    -> {data_out.dtype}")
    return data_out


# # from cellprofiler / centrosome / smooth.py
# def smooth_with_function_and_mask(image, function, mask):
#     """Smooth an image with a linear function, ignoring the contribution of masked pixels

#     image - image to smooth
#     function - a function that takes an image and returns a smoothed image
#     mask  - mask with 1's for significant pixels, 0 for masked pixels

#     This function calculates the fractional contribution of masked pixels
#     by applying the function to the mask (which gets you the fraction of
#     the pixel data that's due to significant points). We then mask the image
#     and apply the function. The resulting values will be lower by the bleed-over
#     fraction, so you can recalibrate by dividing by the function on the mask
#     to recover the effect of smoothing from just the significant pixels.
#     """

#     not_mask = np.logical_not(mask)
#     bleed_over = function(mask.astype(float))
#     masked_image = np.zeros(image.shape, image.dtype)
#     masked_image[mask] = image[mask]
#     smoothed_image = function(masked_image)
#     output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
#     return output_image

#     # blurred_image = centrosome.smooth.smooth_with_function_and_mask(
#     #     data,
#     #     lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
#     #     mask,
#     # )

import numpy as np
from skimage.filters import threshold_triangle, threshold_otsu, threshold_li, threshold_multiotsu, threshold_sauvola

# from skimage.filters import threshold_triangle, threshold_otsu, threshold_li, threshold_multiotsu, threshold_sauvola
from scipy.ndimage import median_filter, extrema
import scipy
from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from typing import Tuple

from skimage.morphology import remove_small_objects


def log_transform(image: np.ndarray) -> np.ndarray:
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


def threshold_li_log(image_in):
    """
    thin wrapper to log-scale and inverse log image for threshold finding using li minimum cross-entropy
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
    """

    image, d = log_transform(image_in.copy())
    threshold = threshold_otsu(image)
    threshold = inverse_log_transform(threshold, d)
    return threshold


def threshold_multiotsu_log(image_in):
    """
    thin wrapper to log-scale and inverse log image for threshold finding using otsu
    """
    image, d = log_transform(image_in.copy())
    thresholds = threshold_multiotsu(image)
    thresholds = inverse_log_transform(thresholds, d)
    return thresholds


# these are the fuinctions that need to be set by each notebook...
# notebook workflow will produce the in_params dictionary nescessary
# so all the images can be pushed through these functions (procedures)

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

    # this might be faster:  scipy.signal.medfilt2d()
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


def find_optimal_Z(raw_img: np.ndarray, nuc_ch: int, ch_to_agg: Tuple[int]) -> int:
    """
    Procedure to infer "optimal" Z-slice  from linearly unmixed input.

    Parameters:
    ------------
    raw_img: np.ndarray
        a ch,z,x,y - image containing florescent signal

    nuc_ch: int
        holds the needed parameters

    ch_to_agg: int
        holds the needed parameters

    Returns:
    -------------
    opt_z:
        the "0ptimal" z-slice which has the most signal intensity for downstream 2D segmentation
    """

    # median filter in 2D / convert to float 0-1.   get rid of the "residual"
    nuclei = simple_intensity_normalization(raw_img[nuc_ch].copy())

    med_filter_size = 4
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    nuclei = median_filter_slice_by_slice(nuclei, size=med_filter_size)

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    nuclei = image_smoothing_gaussian_slice_by_slice(
        nuclei, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )

    # struct_obj = struct_img > filters.threshold_li(struct_img)
    threshold_value_log = threshold_li_log(nuclei)

    threshold_factor = 0.9  # from cellProfiler
    thresh_min = 0.1
    thresh_max = 1.0
    threshold = min(max(threshold_value_log * threshold_factor, thresh_min), thresh_max)

    struct_obj = nuclei > threshold
    # find the z with the maximum non-nuclei florescence
    total_florescence = (
        raw_img[
            ch_to_agg,
        ]
        .astype(np.double)
        .sum(axis=0)
    )
    total_florescence[struct_obj] = 0
    optimal_Z = total_florescence.sum(axis=(1, 2)).argmax()

    return optimal_Z


def size_filter_2D(img: np.ndarray, min_size: int, connectivity: int = 1):
    """size filter

    Parameters:
    ------------
    img: np.ndarray
        the image to filter on
    min_size: int
        the minimum size to keep
    connnectivity: int
        the connectivity to use when computing object size
    """
    return remove_small_objects(img > 0, min_size=min_size, connectivity=connectivity, in_place=False)


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


# # takein from cellprofiler / centrosome
# # since i'm not limiting myself to integers it might not work...
# def fill_labeled_holes(labels, mask=None, size_fn=None):
#     """Fill all background pixels that are holes inside the foreground

#     A pixel is a hole inside a foreground object if

#     * there is no path from the pixel to the edge AND

#     * there is no path from the pixel to any other non-hole
#       pixel AND

#     * there is no path from the pixel to two similarly-labeled pixels that
#       are adjacent to two differently labeled non-hole pixels.

#     labels - the current labeling

#     mask - mask of pixels to ignore

#     size_fn - if not None, it is a function that takes a size and a boolean
#               indicating whether it is foreground (True) or background (False)
#               The function should return True to analyze and False to ignore

#     returns a filled copy of the labels matrix
#     """
#     #
#     # The algorithm:
#     #
#     # Label the background to get distinct background objects
#     # Construct a graph of both foreground and background objects.
#     # Walk the graph according to the rules.
#     #
#     labels_type = labels.dtype
#     background = labels == 0
#     if mask is not None:
#         background &= mask
#     four_connect = 0 # what is this?

#     blabels, count = ndi.label(background, four_connect)
#     labels = labels.copy().astype(int)
#     lcount = np.max(labels)
#     labels[blabels != 0] = blabels[blabels != 0] + lcount + 1
#     lmax = lcount + count + 1
#     is_not_hole = np.ascontiguousarray(np.zeros(lmax + 1, np.uint8))
#     #
#     # Find the indexes on the edge and use to populate the to-do list
#     #
#     to_do = np.unique(
#         np.hstack((labels[0, :], labels[:, 0], labels[-1, :], labels[:, -1]))
#     )
#     to_do = to_do[to_do != 0]
#     is_not_hole[to_do] = True
#     to_do = list(to_do)
#     #
#     # An array that names the first non-hole object
#     #
#     adjacent_non_hole = np.ascontiguousarray(np.zeros(lmax + 1), np.uint32)
#     #
#     # Find all 4-connected adjacent pixels
#     # Note that there will be some i, j not in j, i
#     #
#     i = np.hstack([labels[:-1, :].flatten(), labels[:, :-1].flatten()])
#     j = np.hstack([labels[1:, :].flatten(), labels[:, 1:].flatten()])
#     i, j = i[i != j], j[i != j]
#     if (len(i)) > 0:
#         order = np.lexsort((j, i))
#         i = i[order]
#         j = j[order]
#         # Remove duplicates and stack to guarantee that j, i is in i, j
#         first = np.hstack([[True], (i[:-1] != i[1:]) | (j[:-1] != j[1:])])
#         i, j = np.hstack((i[first], j[first])), np.hstack((j[first], i[first]))
#         # Remove dupes again. (much shorter)
#         order = np.lexsort((j, i))
#         i = i[order]
#         j = j[order]
#         first = np.hstack([[True], (i[:-1] != i[1:]) | (j[:-1] != j[1:])])
#         i, j = i[first], j[first]
#         #
#         # Now we make a ragged array of i and j
#         #
#         i_count = np.bincount(i)
#         if len(i_count) < lmax + 1:
#             i_count = np.hstack((i_count, np.zeros(lmax + 1 - len(i_count), int)))
#         indexer = Indexes([i_count])
#         #
#         # Filter using the size function passed, if any
#         #
#         if size_fn is not None:
#             areas = np.bincount(labels.flatten())
#             for ii, area in enumerate(areas):
#                 if (
#                     ii > 0
#                     and area > 0
#                     and not is_not_hole[ii]
#                     and not size_fn(area, ii <= lcount)
#                 ):
#                     is_not_hole[ii] = True
#                     to_do.append(ii)

#         to_do_count = len(to_do)
#         if len(to_do) < len(is_not_hole):
#             to_do += [0] * (len(is_not_hole) - len(to_do))
#         to_do = np.ascontiguousarray(np.array(to_do), np.uint32)
#         fill_labeled_holes_loop(
#                                     np.ascontiguousarray(i, np.uint32),
#                                     np.ascontiguousarray(j, np.uint32),
#                                     np.ascontiguousarray(indexer.fwd_idx, np.uint32),
#                                     np.ascontiguousarray(i_count, np.uint32),
#                                     is_not_hole,
#                                     adjacent_non_hole,
#                                     to_do,
#                                     lcount,
#                                     to_do_count,
#                                 )
#     #
#     # Make an array that assigns objects to themselves and background to 0
#     #
#     new_indexes = np.arange(len(is_not_hole)).astype(np.uint32)
#     new_indexes[(lcount + 1) :] = 0
#     #
#     # Fill the holes by replacing the old value by the value of the
#     # enclosing object.
#     #
#     is_not_hole = is_not_hole.astype(bool)
#     new_indexes[~is_not_hole] = adjacent_non_hole[~is_not_hole]
#     if mask is not None:
#         labels[mask] = new_indexes[labels[mask]]
#     else:
#         labels = new_indexes[labels]
#     return labels.astype(labels_type)


TS_GLOBAL = "Global"
TS_ADAPTIVE = "Adaptive"
TM_MANUAL = "Manual"
TM_MEASUREMENT = "Measurement"
TM_LI = "Minimum Cross-Entropy"
TM_OTSU = "Otsu"
TM_ROBUST_BACKGROUND = "Robust Background"
TM_SAUVOLA = "Sauvola"

# GET_GLOBAL_THRESHOLD
def get_global_threshold(
    image, mask, threshold_operation=TM_LI, automatic=False, two_class_otsu=False, assign_mid_to_fg=True
):

    image_data = image[mask]
    # Shortcuts - Check if image array is empty or all pixels are the same value.
    if len(image_data) == 0:
        threshold = 0.0

    elif np.all(image_data == image_data[0]):
        threshold = image_data[0]

    elif automatic or threshold_operation in (TM_LI, TM_SAUVOLA):
        # tol = max(np.min(np.diff(np.unique(image_data))) / 2, 0.5 / 65536)
        threshold = threshold_li(image_data)  # , tolerance=tol)

    elif threshold_operation == TM_OTSU:
        if two_class_otsu:
            threshold = threshold_otsu(image_data)
        else:
            bin_wanted = 0 if assign_mid_to_fg else 1
            threshold = threshold_multiotsu(image_data, nbins=128)
            threshold = threshold[bin_wanted]
    else:
        raise ValueError("Invalid thresholding settings")
    return threshold


# GET_LOCAL_THRESHOLD
def get_local_threshold(
    image, mask, volumetric, adaptive_window_size=40, threshold_operation=TM_LI, two_class_otsu=False
):

    image_data = np.where(mask, image, np.nan)

    if len(image_data) == 0 or np.all(image_data == np.nan):
        local_threshold = np.zeros_like(image_data)

    elif np.all(image_data == image_data[0]):
        local_threshold = np.full_like(image_data, image_data[0])

    elif threshold_operation == TM_LI:
        local_threshold = _run_local_threshold(
            image_data,
            method=threshold_li,
            volumetric=volumetric,
            # tolerance=max(np.min(np.diff(np.unique(image))) / 2, 0.5 / 65536)
        )
    elif threshold_operation == TM_OTSU:
        if two_class_otsu:
            local_threshold = _run_local_threshold(
                image_data,
                method=threshold_otsu,
                volumetric=volumetric,
            )
        else:
            local_threshold = _run_local_threshold(
                image_data,
                method=threshold_multiotsu,
                volumetric=volumetric,
                nbins=128,
            )

    elif threshold_operation == TM_SAUVOLA:
        image_data = np.where(mask, image, 0)
        adaptive_window = adaptive_window_size
        if adaptive_window % 2 == 0:
            adaptive_window += 1
        local_threshold = threshold_sauvola(image_data, window_size=adaptive_window)

    else:
        raise ValueError("Invalid thresholding settings")
    return local_threshold


# RUN_GLOBAL_THRESHOLD
def _run_local_threshold(
    image_data,
    method,
    volumetric,
    threshold_operation=TM_LI,
    automatic=False,
    two_class_otsu=False,
    assign_mid_to_fg=True,
    adaptive_window_size=80,
):
    if volumetric:
        t_local = np.zeros_like(image_data)
        for index, plane in enumerate(image_data):
            t_local[index] = _get_adaptive_threshold(
                plane,
                method,
                threshold_operation=threshold_operation,
                automatic=automatic,
                two_class_otsu=two_class_otsu,
                assign_mid_to_fg=assign_mid_to_fg,
                adaptive_window_size=adaptive_window_size,
            )
    else:
        t_local = _get_adaptive_threshold(
            image_data,
            method,
            threshold_operation=threshold_operation,
            automatic=automatic,
            two_class_otsu=two_class_otsu,
            assign_mid_to_fg=assign_mid_to_fg,
            adaptive_window_size=adaptive_window_size,
        )
    retval = np.img_as_float(t_local)
    return retval


# _GET_ADAPTIVE_THRESHOLD
def _get_adaptive_threshold(
    image_data,
    threshold_method,
    threshold_operation=TM_LI,
    automatic=False,
    two_class_otsu=False,
    assign_mid_to_fg=True,
    adaptive_window_size=80,
):
    """Given a global threshold, compute a threshold per pixel
    Break the image into blocks, computing the threshold per block.
    Afterwards, constrain the block threshold to .7 T < t < 1.5 T.
    """
    # for the X and Y direction, find the # of blocks, given the
    # size constraints
    if threshold_operation == TM_OTSU:
        bin_wanted = 0 if assign_mid_to_fg else 1
    image_size = np.array(image_data.shape[:2], dtype=int)
    nblocks = image_size // adaptive_window_size
    if any(n < 2 for n in nblocks):
        raise ValueError(
            "Adaptive window cannot exceed 50%% of an image dimension.\n"
            "Window of %dpx is too large for a %sx%s image" % (adaptive_window_size, image_size[1], image_size[0])
        )
    #
    # Use a floating point block size to apportion the roundoff
    # roughly equally to each block
    #
    increment = np.array(image_size, dtype=float) / np.array(nblocks, dtype=float)
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
            elif threshold_operation == TM_OTSU and two_class_otsu and len(np.unique(block)) < 3:
                # Can't run 3-class otsu on only 2 values.
                threshold_out = threshold_otsu(block)
            else:
                try:
                    threshold_out = threshold_method(block)
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
    thresh_out_x_coords = np.linspace(0.5, int(nblocks[0] * increment[0]) - 0.5, thresh_out.shape[0])
    thresh_out_y_coords = np.linspace(0.5, int(nblocks[1] * increment[1]) - 0.5, thresh_out.shape[1])

    thresh_out = adaptive_interpolation(thresh_out_x_coords, thresh_out_y_coords)

    return thresh_out


def _correct_global_threshold(threshold, corr_value, threshold_range):
    threshold *= corr_value
    return min(max(threshold, threshold_range.min), threshold_range.max)


def _correct_local_threshold(t_local_orig, t_guide, threshold_correction_factor, threshold_range):
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
    image_data, mask, th_method=TM_LI, volumetric=True, window_size=40, log_scale=False  # skimage.filters.threshold_li,
):
    """
    wrapper for the functions from CellProfiler
    NOTE: might work better to copy from CellProfiler/centrosome/threshold.py
    https://github.com/CellProfiler/centrosome/blob/master/centrosome/threshold.py
    """

    th_guide = get_global_threshold(image_data, mask, threshold_operation=th_method)

    th_original = get_local_threshold(
        image_data, mask, volumetric=volumetric, adaptive_window_size=window_size, threshold_operation=th_method
    )

    final_threshold, orig_threshold, guide_threshold = get_threshold(
        image_data,
        mask,
        th_guide,
        th_original,
        threshold_operation=th_method,
        volumetric=volumetric,
        log_scale=log_scale,
        threshold_scope=TS_ADAPTIVE,
    )

    binary_image, _ = apply_threshold(image_data, final_threshold)
    return binary_image


def apply_threshold(image, threshold, mask=None, automatic=False):
    if mask is not None:
        return (image >= threshold) & mask
    else:
        return image >= threshold


def get_threshold(
    image,
    mask,
    th_guide,
    th_original,
    threshold_operation=TM_LI,
    volumetric=True,
    automatic=False,
    log_scale=False,
    threshold_scope=TS_GLOBAL,
):

    need_transform = threshold_operation in (TM_LI, TM_OTSU) and log_scale

    if need_transform:
        image_data, conversion_dict = log_transform(image)
    else:
        image_data = image

    if threshold_scope == TS_GLOBAL or automatic:
        th_guide = None
        th_original = get_global_threshold(image_data, mask, automatic=automatic)

    elif threshold_scope == TS_ADAPTIVE:
        th_guide = get_global_threshold(image_data, mask)
        th_original = get_local_threshold(image_data, mask, volumetric)
    else:
        raise ValueError("Invalid thresholding settings")

    if need_transform:
        th_original = inverse_log_transform(th_original, conversion_dict)
        if th_guide is not None:
            th_guide = inverse_log_transform(th_guide, conversion_dict)

    # apply correction
    if threshold_scope == TS_GLOBAL or automatic:
        th_corrected = _correct_global_threshold(th_original)
    else:
        th_guide = _correct_global_threshold(th_guide)
        th_corrected = _correct_local_threshold(th_original, th_guide)

    return th_corrected, th_original, th_guide


def get_threshold_robust_background(image_data, num_dev=3.0, lower_fraction=0.05, upper_fraction=0.05):
    """
    derived from cell-profiler... using for lipid body segmentation.
    """

    flat_image = image_data.flatten()
    n_pixels = len(flat_image)
    if n_pixels < 3:
        return 0

    flat_image.sort()
    if flat_image[0] == flat_image[-1]:
        return flat_image[0]
    low_chop = int(round(n_pixels * lower_fraction))
    hi_chop = n_pixels - int(round(n_pixels * upper_fraction))
    im = flat_image if low_chop == 0 else flat_image[low_chop:hi_chop]
    mean = np.mean(im)
    sd = np.std(im)
    return mean + sd * num_dev


# from cellprofiler / centrosome / smooth.py
def smooth_with_function_and_mask(image, function, mask):
    """Smooth an image with a linear function, ignoring the contribution of masked pixels

    image - image to smooth
    function - a function that takes an image and returns a smoothed image
    mask  - mask with 1's for significant pixels, 0 for masked pixels

    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the bleed-over
    fraction, so you can recalibrate by dividing by the function on the mask
    to recover the effect of smoothing from just the significant pixels.
    """

    not_mask = np.logical_not(mask)
    bleed_over = function(mask.astype(float))
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image

    # blurred_image = centrosome.smooth.smooth_with_function_and_mask(
    #     data,
    #     lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
    #     mask,
    # )

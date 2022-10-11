"""
Transforms are wrappers to filters which manipulate the "image" in place and keep track of 
which filters/ segmentations

"""


from abc import ABC
import cv2
import numpy as np
import pandas as pd

from skimage import restoration
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity
from skimage.measure import regionprops_table

from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    image_smoothing_gaussian_3d,
    image_smoothing_gaussian_slice_by_slice,
)

# this is causing a circular import... and it is unnescessaruy... we can just use a protocol
# from .object import BioImObject
# from ..bioim.image import BioImImage as BioImIm
from ..bioim.base import BioImContainer, MetaArrayLike

# use ABC to prevent circular import issues
from abc import ABC, abstractmethod
from typing import Any

# import pathml.core
# import pathml.core.slide_data

# Base class


class BioImTransformBase(ABC):
    """
    Base class for all Transforms.
    Each transform must operate on a Tile.
    """

    @abstractmethod
    def F(self, im_array: MetaArrayLike) -> MetaArrayLike:
        ...

    @abstractmethod
    def apply(self, img: Any):
        ...


class BioImTransform(BioImTransformBase):
    """
    Base class for all Transforms.
    Each transform must operate on a Tile.
    """

    def __repr__(self):
        return "Base class for all transforms"

    def F(self, im_array: MetaArrayLike) -> MetaArrayLike:
        """functional implementation"""
        raise NotImplementedError

    def apply(self, img: BioImContainer):
        """modify Image object in-place"""
        #     raise NotImplementedError
        # def apply(self, img):
        assert isinstance(img, BioImContainer), f"img is type {type(img)} but must be Image"
        # img.set_image(self.F(img.get_image()))
        img.data = self.F(img.data)
        img.add_transform(self)


# implement transforms here


class MedianBlurSliceBySlice(BioImTransform):
    """
    Median blur kernel.
    Args:
        kernel_size (int): Width of kernel. Must be an odd number. Defaults to 5.
    """

    def __init__(self, kernel_size=5):
        assert kernel_size % 2 == 1, "kernel_size must be an odd number"
        self.kernel_size = kernel_size

    def __repr__(self):
        return f"MedianBlurSliceBySlice(kernel_size={self.kernel_size})"

    def F(self, image: MetaArrayLike) -> MetaArrayLike:
        # assert image.dtype == np.uint8, f"image dtype {image.dtype} must be np.uint8"
        out = median_filter_slice_by_slice(image, kernel_size)
        return out
        # return cv2.medianBlur(image, ksize=self.kernel_size)

    # def apply(self, img):
    #     # assert isinstance(
    #     #     img, BioImImage
    #     # ), f"img is type {type(img)} but must be Image"
    #     img.set_image = self.F(img.get_image())


class GaussianBlurSliceBySlice(BioImTransform):
    """
    Gaussian blur kernel.
    Args:
        truncate_range (float): Width of kernel. in SDs.Defaults to 3.0.
        sigma (float): width of Gaussian kernel: assumed to be equal in X and Y axes. Defaults to 1.34 (e/2)
    """

    def __init__(self, truncate_range=3.0, sigma=1.34):
        self.truncate_range = truncate_range
        self.sigma = sigma

    def __repr__(self) -> str:
        return f"GaussianBlurSliceBySlice(truncate_range={self.truncate_range}, sigma={self.sigma})"

    def F(self, image: MetaArrayLike) -> MetaArrayLike:
        # assert image.dtype == np.uint8, f"image dtype {image.dtype} must be np.uint8"
        out = image_smoothing_gaussian_slice_by_slice(image, sigma=self.sigma, truncate_range=self.truncate_range)
        return out


class BoxBlur(BioImTransform):
    """
    Box (average) blur kernel.
    Args:
        kernel_size (int): Width of kernel. Defaults to 5.
    """

    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def __repr__(self):
        return f"BoxBlur(kernel_size={self.kernel_size})"

    def F(self, image: MetaArrayLike) -> MetaArrayLike:
        assert image.dtype == np.uint8, f"image dtype {image.dtype} must be np.uint8"
        return cv2.boxFilter(image, ksize=(self.kernel_size, self.kernel_size), ddepth=-1)


class RescaleIntensity(BioImTransform):
    """
    Return image after stretching or shrinking its intensity levels.
    The desired intensity range of the input and output, in_range and out_range respectively, are used to stretch or shrink the intensity range of the input image
    This function is a wrapper for 'rescale_intensity' function from scikit-image: https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    Args:
        in_range (str or 2-tuple, optional): Min and max intensity values of input image. The possible values for this parameter are enumerated below.
            ‘image’ : Use image min/max as the intensity range.
            ‘dtype’ : Use min/max of the image’s dtype as the intensity range.
            'dtype-name' : Use intensity range based on desired dtype. Must be valid key in DTYPE_RANGE.
            '2-tuple' : Use range_values as explicit min/max intensities.
        out_range (str or 2-tuple, optional): Min and max intensity values of output image. The possible values for this parameter are enumerated below.
            ‘image’ : Use image min/max as the intensity range.
            ‘dtype’ : Use min/max of the image’s dtype as the intensity range.
            'dtype-name' : Use intensity range based on desired dtype. Must be valid key in DTYPE_RANGE.
            '2-tuple' : Use range_values as explicit min/max intensities.
    """

    def __init__(self, in_range="image", out_range="dtype"):
        self.in_range = in_range
        self.out_range = out_range

    def __repr__(self):
        return f"RescaleIntensity(in_range={self.in_range}, out_range={self.out_range})"

    def F(self, image: MetaArrayLike) -> MetaArrayLike:
        image = rescale_intensity(image, in_range=self.in_range, out_range=self.out_range)
        return image


class HistogramEqualization(BioImTransform):
    """
     Return image after histogram equalization.
     This function is a wrapper for 'equalize_hist' function from scikit-image: https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    Args:
        nbins (int, optional): Number of gray bins for histogram. Note: this argument is ignored for integer images, for which each integer is its own bin.
        mask (ndarray of bools or 0s and 1s, optional): Array of same shape as image. Only points at which mask == True are used for the equalization, which is applied to the whole image.
    """

    def __init__(self, nbins=256, mask=None):
        self.nbins = nbins
        self.mask = mask

    def __repr__(self):
        return f"HistogramEqualization(nbins={self.nbins}, mask = {self.mask})"

    def F(self, image: MetaArrayLike) -> MetaArrayLike:
        image = equalize_hist(image, nbins=self.nbins, mask=self.mask)
        return image


class AdaptiveHistogramEqualization(BioImTransform):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    This function is a wrapper for 'equalize_adapthist' function from scikit-image: https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
    Args:
        kernel_size (int or array_like, optional): Defines the shape of contextual regions used in the algorithm. If iterable is passed, it must have the same number of elements as image.ndim (without color channel). If integer, it is broadcasted to each image dimension. By default, kernel_size is 1/8 of image height by 1/8 of its width.
        clip_limit (float): Clipping limit, normalized between 0 and 1 (higher values give more contrast).
        nbins (int): Number of gray bins for histogram (“data range”).
    """

    def __init__(self, kernel_size=None, clip_limit=0.3, nbins=256):
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.nbins = nbins

    def __repr__(self):
        return f"AdaptiveHistogramEqualization(kernel_size={self.kernel_size}, clip_limit={self.clip_limit}, nbins={self.nbins})"

    def F(self, image: MetaArrayLike) -> MetaArrayLike:
        image = equalize_adapthist(
            image,
            kernel_size=self.kernel_size,
            clip_limit=self.clip_limit,
            nbins=self.nbins,
        )
        return image


# class BinaryThreshold(BioImTransform):
#     """
#     Binary thresholding transform to create a binary mask.
#     If input image is RGB it is first converted to greyscale, otherwise the input must have 1 channel.
#     Args:
#         mask_name (str): Name of mask that is created.
#         use_otsu (bool): Whether to use Otsu's method to automatically determine optimal threshold. Defaults to True.
#         threshold (int): Specified threshold. Ignored if ``use_otsu is True``. Defaults to 0.
#         inverse (bool): Whether to use inverse threshold. If using inverse threshold, pixels below the threshold will
#             be returned as 1. Otherwise pixels below the threshold will be returned as 0. Defaults to ``False``.
#     References:
#         Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE transactions on systems,
#         man, and cybernetics, 9(1), pp.62-66.
#     """

#     def __init__(self, mask_name=None, use_otsu=True, threshold=0, inverse=False):
#         self.threshold = threshold
#         self.max_value = 255
#         self.use_otsu = use_otsu
#         self.inverse = inverse
#         self.mask_name = mask_name
#         self.type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
#         if use_otsu:
#             self.type += cv2.THRESH_OTSU

#     def __repr__(self):
#         return (
#             f"BinaryThreshold(use_otsu={self.use_otsu}, threshold={self.threshold}, "
#             f"mask_name={self.mask_name}, inverse={self.inverse})"
#         )

#     def F(self, image: MetaArrayLike) -> MetaArrayLike:
#         assert image.dtype == np.uint8, f"image dtype {image.dtype} must be np.uint8"
#         assert image.ndim == 2, f"input image has shape {image.shape}. Must convert to 1-channel image (H, W)."
#         _, out = cv2.threshold(
#             src=image,
#             thresh=self.threshold,
#             maxval=self.max_value,
#             type=self.type,
#         )
#         return out.astype(np.uint8)

#     def apply(self, img):
#         # assert isinstance(
#         #     img, Image
#         # ), f"img is type {type(img)} but must be Image"
#         im = img.image

#         assert self.mask_name is not None, f"mask_name is None. Must supply a valid mask name"
#         # assert im.ndim == 2, "chunk.image is not RGB and has more than 1 channel"
#         thresholded_mask = self.F(im)
#         img.mask = thresholded_mask
#         # tile.masks[self.mask_name] = thresholded_mask


# class MorphOpen(BioImTransform):
#     """
#     Morphological opening. First applies erosion operation, then dilation.
#     Reduces noise by removing small objects from the background.
#     Operates on a binary mask.
#     Args:
#         kernel_size (int): Size of kernel for default square kernel. Ignored if a custom kernel is specified.
#             Defaults to 5.
#         n_iterations (int): Number of opening operations to perform. Defaults to 1.
#     """

#     def __init__(self, mask_name=None, kernel_size=5, n_iterations=1):
#         self.kernel_size = kernel_size
#         self.n_iterations = n_iterations
#         self.mask_name = mask_name

#     def __repr__(self):
#         return (
#             f"MorphOpen(kernel_size={self.kernel_size}, n_iterations={self.n_iterations}, "
#             f"mask_name={self.mask_name})"
#         )

#     def F(self, mask):
#         assert mask.dtype == np.uint8, f"mask type {mask.dtype} must be np.uint8"
#         k = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
#         out = cv2.morphologyEx(src=mask, kernel=k, op=cv2.MORPH_OPEN, iterations=self.n_iterations)
#         return out

#     def apply(self, tile):
#         # assert isinstance(
#         #     image, Image
#         # ), f"image is type {type(image)} but must be BioImIm"
#         assert self.mask_name is not None, "mask_name is None. Must supply a valid mask name"
#         m = np.copy(tile.masks[self.mask_name])
#         out = self.F(m)
#         tile.masks[self.mask_name] = out


# class MorphClose(BioImTransform):
#     """
#     Morphological closing. First applies dilation operation, then erosion.
#     Reduces noise by closing small holes in the foreground.
#     Operates on a binary mask.
#     Args:
#         mask_name (str): Name of mask on which to apply transform
#         kernel_size (int): Size of kernel for default square kernel. Ignored if a custom kernel is specified.
#             Defaults to 5.
#         n_iterations (int): Number of opening operations to perform. Defaults to 1.
#     """

#     def __init__(self, mask_name=None, kernel_size=5, n_iterations=1):
#         self.kernel_size = kernel_size
#         self.n_iterations = n_iterations
#         self.mask_name = mask_name

#     def __repr__(self):
#         return (
#             f"MorphClose(kernel_size={self.kernel_size}, n_iterations={self.n_iterations}, "
#             f"mask_name={self.mask_name})"
#         )

#     def F(self, mask):
#         assert mask.dtype == np.uint8, f"mask type {mask.dtype} must be np.uint8"
#         k = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
#         out = cv2.morphologyEx(src=mask, kernel=k, op=cv2.MORPH_CLOSE, iterations=self.n_iterations)
#         return out

#     def apply(self, tile):
#         assert self.mask_name is not None, "mask_name is None. Must supply a valid mask name"
#         m = np.copy(tile.masks[self.mask_name])
#         out = self.F(m)
#         tile.masks[self.mask_name] = out


# class ForegroundDetection(BioImTransform):
#     """
#     Foreground detection for binary masks. Identifies regions that have a total area greater than
#     specified threshold. Supports including holes within foreground regions, or excluding holes
#     above a specified area threshold.
#     Args:
#         min_region_size (int): Minimum area of detected foreground regions, in pixels. Defaults to 5000.
#         max_hole_size (int): Maximum size of allowed holes in foreground regions, in pixels.
#             Ignored if ``outer_contours_only is True``. Defaults to 1500.
#         outer_contours_only (bool): If true, ignore holes in detected foreground regions. Defaults to False.
#         mask_name (str): Name of mask on which to apply transform
#     References:
#         Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M. and Mahmood, F., 2020. Data Efficient and
#         Weakly Supervised Computational Pathology on Whole Slide Images. arXiv preprint arXiv:2004.09666.
#     """

#     def __init__(
#         self,
#         mask_name=None,
#         min_region_size=5000,
#         max_hole_size=1500,
#         outer_contours_only=False,
#     ):
#         self.min_region_size = min_region_size
#         self.max_hole_size = max_hole_size
#         self.outer_contours_only = outer_contours_only
#         self.mask_name = mask_name

#     def __repr__(self):
#         return (
#             f"ForegroundDetection(min_region_size={self.min_region_size}, max_hole_size={self.max_hole_size},"
#             f"outer_contours_only={self.outer_contours_only}, mask_name={self.mask_name})"
#         )

#     def F(self, mask):
#         assert mask.dtype == np.uint8, f"mask type {mask.dtype} must be np.uint8"
#         mode = cv2.RETR_EXTERNAL if self.outer_contours_only else cv2.RETR_CCOMP
#         contours, hierarchy = cv2.findContours(mask.copy(), mode=mode, method=cv2.CHAIN_APPROX_NONE)

#         if hierarchy is None:
#             # no contours found --> return empty mask
#             mask_out = np.zeros_like(mask)
#         elif self.outer_contours_only:
#             out = np.zeros_like(mask, dtype=np.int8)
#             for c in contours:
#                 # ignore contours below size threshold
#                 if cv2.contourArea(c) > self.min_region_size:
#                     # fill contours
#                     cv2.fillPoly(out, [c], 255)
#             mask_out = out
#         else:
#             # separate outside and inside contours (region boundaries vs. holes in regions)
#             # find the outside contours by looking for those with no parents (4th column is -1 if no parent)
#             hierarchy = np.squeeze(hierarchy, axis=0)
#             outside_contours = hierarchy[:, 3] == -1
#             hole_contours = ~outside_contours

#             # outside contours must be above min_tissue_region_size threshold
#             contour_size_thresh = [cv2.contourArea(c) > self.min_region_size for c in contours]
#             # hole contours must be above area threshold
#             hole_size_thresh = [cv2.contourArea(c) > self.max_hole_size for c in contours]
#             # holes must have parents above area threshold
#             hole_parent_thresh = [p in np.argwhere(contour_size_thresh).flatten() for p in hierarchy[:, 3]]

#             outside_contours = np.array(outside_contours)
#             hole_contours = np.array(hole_contours)
#             contour_size_thresh = np.array(contour_size_thresh)
#             hole_size_thresh = np.array(hole_size_thresh)
#             hole_parent_thresh = np.array(hole_parent_thresh)

#             # now combine outside and inside contours into final mask
#             out1 = np.zeros_like(mask, dtype=np.int8)
#             out2 = np.zeros_like(mask, dtype=np.int8)

#             # loop thru contours
#             for (cnt, outside, size_thresh, hole, hole_size_thresh, hole_parent_thresh,) in zip(
#                 contours,
#                 outside_contours,
#                 contour_size_thresh,
#                 hole_contours,
#                 hole_size_thresh,
#                 hole_parent_thresh,
#             ):
#                 if outside and size_thresh:
#                     # in this case, the contour is an outside contour
#                     cv2.fillPoly(out1, [cnt], 255)
#                 if hole and hole_size_thresh and hole_parent_thresh:
#                     # in this case, the contour is an inside contour
#                     cv2.fillPoly(out2, [cnt], 255)

#             mask_out = out1 - out2

#         return mask_out.astype(np.uint8)

#     def apply(self, image):
#         # assert isinstance(image, BioImIm), f"image is type {type(image)} but must be BioImIm"
#         assert self.mask_name is not None, "mask_name is None. Must supply a valid mask name"
#         m = image.masks[self.mask_name]
#         mask_out = self.F(m)
#         image.masks[self.mask_name] = mask_out


class SuperpixelInterpolation(BioImTransform):
    """
    Divide input image into superpixels using SLIC algorithm, then interpolate each superpixel with average color.
    SLIC superpixel algorithm described in Achanta et al. 2012.
    Args:
        region_size (int): region_size parameter used for superpixel creation. Defaults to 10.
        n_iter (int): Number of iterations to run SLIC algorithm. Defaults to 30.
    References:
        Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P. and Süsstrunk, S., 2012. SLIC superpixels compared to
        state-of-the-art superpixel methods. IEEE transactions on pattern analysis and machine intelligence, 34(11),
        pp.2274-2282.
    """

    def __init__(self, region_size=10, n_iter=30):
        self.region_size = region_size
        self.n_iter = n_iter

    def __repr__(self):
        return f"SuperpixelInterpolation(region_size={self.region_size}, n_iter={self.n_iter})"

    def F(self, image: MetaArrayLike) -> MetaArrayLike:
        assert image.dtype == np.uint8, f"Input image dtype {image.dtype} must be np.uint8"
        # initialize slic class and iterate
        slic = cv2.ximgproc.createSuperpixelSLIC(image=image, region_size=self.region_size)
        slic.iterate(num_iterations=self.n_iter)
        labels = slic.getLabels()
        n_labels = slic.getNumberOfSuperpixels()
        out = image.copy()
        # TODO apply over channels instead of looping
        for i in range(n_labels):
            mask = labels == i
            for c in range(3):
                av = np.mean(image[:, :, c][mask])
                out[:, :, c][mask] = av
        return out


# class BespokeBioImTransformExample(BioImTransform):
#     """
#     Normalize H&E stained images to a reference slide.
#     Also can be used to separate hematoxylin and eosin channels.
#     H&E images are assumed to be composed of two stains, each one having a vector of its characteristic RGB values.
#     The stain matrix is a 3x2 matrix where the first column corresponds to the hematoxylin stain vector and the second
#     corresponds to eosin stain vector. The stain matrix can be estimated from a reference image in a number of ways;
#     here we provide implementations of two such algorithms from Macenko et al. and Vahadane et al.
#     After estimating the stain matrix for an image, the next step is to assign stain concentrations to each pixel.
#     Each pixel is assumed to be a linear combination of the two stain vectors, where the coefficients are the
#     intensities of each stain vector at that pixel. To solve for the intensities, we use least squares in Macenko
#     method and lasso in vahadane method.
#     The image can then be reconstructed by applying those pixel intensities to a stain matrix. This allows you to
#     standardize the appearance of an image by reconstructing it using a reference stain matrix. Using this method of
#     normalization may help account for differences in slide appearance arising from variations in staining procedure,
#     differences between scanners, etc. Images can also be reconstructed using only a single stain vector, e.g. to
#     separate the hematoxylin and eosin channels of an H&E image.
#     This code is based in part on StainTools: https://github.com/Peter554/StainTools
#     Args:
#         target (str): one of 'normalize', 'hematoxylin', or 'eosin'. Defaults to 'normalize'
#         stain_estimation_method (str): method for estimating stain matrix. Must be one of 'macenko' or 'vahadane'.
#             Defaults to 'macenko'.
#         optical_density_threshold (float): Threshold for removing low-optical density pixels when estimating stain
#             vectors. Defaults to 0.15
#         sparsity_regularizer (float): Regularization parameter for dictionary learning when estimating stain vector
#             using vahadane method. Ignored if ``concentration_estimation_method != 'vahadane'``. Defaults to 1.0
#         angular_percentile (float): Percentile for stain vector selection when estimating stain vector
#             using Macenko method. Ignored if ``concentration_estimation_method != 'macenko'``. Defaults to 0.01
#         regularizer_lasso (float): regularization parameter for lasso solver. Defaults to 0.01.
#             Ignored if ``method != 'lasso'``
#         background_intensity (int): Intensity of background light. Must be an integer between 0 and 255.
#             Defaults to 245.
#         stain_matrix_target_od (np.ndarray): Stain matrix for reference slide.
#             Matrix of H and E stain vectors in optical density (OD) space.
#             Stain matrix is (3, 2) and first column corresponds to hematoxylin.
#             Default stain matrix can be used, or you can also fit to a reference slide of your choosing by calling
#             :meth:`~pathml.preprocessing.transforms.StainNormalizationHE.fit_to_reference`.
#         max_c_target (np.ndarray): Maximum concentrations of each stain in reference slide.
#             Default can be used, or you can also fit to a reference slide of your choosing by calling
#             :meth:`~pathml.preprocessing.transforms.StainNormalizationHE.fit_to_reference`.
#     Note:
#         If using ``stain_estimation_method = "Vahadane"``, `spams <http://thoth.inrialpes.fr/people/mairal/spams/>`_
#         must be installed, along with all of its dependencies (i.e. libblas & liblapack).
#     References:
#         Macenko, M., Niethammer, M., Marron, J.S., Borland, D., Woosley, J.T., Guan, X., Schmitt, C. and Thomas, N.E.,
#         2009, June. A method for normalizing histology slides for quantitative analysis. In 2009 IEEE International
#         Symposium on Biomedical Imaging: From Nano to Macro (pp. 1107-1110). IEEE.
#         Vahadane, A., Peng, T., Sethi, A., Albarqouni, S., Wang, L., Baust, M., Steiger, K., Schlitter, A.M., Esposito,
#         I. and Navab, N., 2016. Structure-preserving color normalization and sparse stain separation for histological
#         images. IEEE transactions on medical imaging, 35(8), pp.1962-1971.
#     """

#     def __init__(
#         self,
#         target="normalize",
#         stain_estimation_method="macenko",
#         optical_density_threshold=0.15,
#         sparsity_regularizer=1.0,
#         angular_percentile=0.01,
#         regularizer_lasso=0.01,
#         background_intensity=245,
#         stain_matrix_target_od=np.array(
#             [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]
#         ),
#         max_c_target=np.array([1.9705, 1.0308]),
#     ):
#         # verify inputs
#         assert target.lower() in [
#             "normalize",
#             "eosin",
#             "hematoxylin",
#         ], f"Error input target {target} must be one of 'normalize', 'eosin', 'hematoxylin'"
#         assert stain_estimation_method.lower() in [
#             "macenko",
#             "vahadane",
#         ], f"Error: input stain estimation method {stain_estimation_method} must be one of 'macenko' or 'vahadane'"
#         assert (
#             0 <= background_intensity <= 255
#         ), f"Error: input background intensity {background_intensity} must be an integer between 0 and 255"

#         if stain_estimation_method.lower() == "vahadane":
#             try:
#                 import spams
#             except (ImportError, ModuleNotFoundError):
#                 raise Exception(
#                     f"Vahadane method requires `spams` package to be installed"
#                 )

#         self.target = target.lower()
#         self.stain_estimation_method = stain_estimation_method.lower()
#         self.optical_density_threshold = optical_density_threshold
#         self.sparsity_regularizer = sparsity_regularizer
#         self.angular_percentile = angular_percentile
#         self.regularizer_lasso = regularizer_lasso
#         self.background_intensity = background_intensity
#         self.stain_matrix_target_od = stain_matrix_target_od
#         self.max_c_target = max_c_target

#     def __repr__(self):
#         return (
#             f"StainNormalizationHE(target={self.target}, stain_estimation_method={self.stain_estimation_method}, "
#             f"optical_density_threshold={self.optical_density_threshold}, "
#             f"sparsity_regularizer={self.sparsity_regularizer}, angular_percentile={self.angular_percentile}, "
#             f"regularizer_lasso={self.regularizer_lasso}, background_intensity={self.background_intensity}, "
#             f"stain_matrix_target_od={self.stain_matrix_target_od}, max_c_target={self.max_c_target})"
#         )

#     def fit_to_reference(self, image_ref):
#         """
#         Fit ``stain_matrix`` and ``max_c`` to a reference slide. This allows you to use a specific slide as the
#         reference for stain normalization. Works by first estimating stain matrix from input reference image,
#         then estimating pixel concentrations. Newly computed stain matrix and maximum concentrations are then used
#         for any future color normalization.
#         Args:
#             image_ref (np.ndarray): RGB reference image
#         """
#         # first estimate stain matrix for reference image_ref
#         stain_matrix = self._estimate_stain_vectors(image=image_ref)

#         # next get pixel concentrations for reference image_ref
#         C = self._estimate_pixel_concentrations(
#             image=image_ref, stain_matrix=stain_matrix
#         )

#         # get max concentrations
#         # actually use 99th percentile so it's more robust
#         max_C = np.percentile(C, 99, axis=0).reshape((1, 2))

#         # put the newly determined stain matrix and max C matrix for reference slide into class attrs
#         self.stain_matrix_target_od = stain_matrix
#         self.max_c_target = max_C

#     def _estimate_stain_vectors(self, image):
#         """
#         Estimate stain vectors using appropriate method
#         Args:
#             image (np.ndarray): RGB image
#         """
#         # first estimate stain matrix for reference image_ref
#         if self.stain_estimation_method == "macenko":
#             stain_matrix = self._estimate_stain_vectors_macenko(image)
#         elif self.stain_estimation_method == "vahadane":
#             stain_matrix = self._estimate_stain_vectors_vahadane(image)
#         else:
#             raise Exception(
#                 f"Error: input stain estimation method {self.stain_estimation_method} must be one of 'macenko' or 'vahadane'"
#             )
#         return stain_matrix

#     def _estimate_pixel_concentrations(self, image, stain_matrix):
#         """
#         Estimate pixel concentrations from a given stain matrix using appropriate method
#         Args:
#             image (np.ndarray): RGB image
#             stain_matrix (np.ndarray): matrix of H and E stain vectors in optical density (OD) space.
#                 Stain_matrix is (3, 2) and first column corresponds to hematoxylin by convention.
#         """
#         if self.stain_estimation_method == "macenko":
#             C = self._estimate_pixel_concentrations_lstsq(image, stain_matrix)
#         elif self.stain_estimation_method == "vahadane":
#             C = self._estimate_pixel_concentrations_lasso(image, stain_matrix)
#         else:
#             raise Exception(f"Provided target {self.target} invalid")
#         return C

#     def _estimate_stain_vectors_vahadane(self, image, random_seed=0):
#         """
#         Estimate stain vectors using dictionary learning method from Vahadane et al.
#         Args:
#             image (np.ndarray): RGB image
#         """
#         try:
#             import spams
#         except (ImportError, ModuleNotFoundError):
#             raise Exception(f"Vahadane method requires `spams` package to be installed")
#         # convert to Optical Density (OD) space
#         image_OD = RGB_to_OD(image)
#         # reshape to (M*N)x3
#         image_OD = image_OD.reshape(-1, 3)
#         # drop pixels with low OD
#         OD = image_OD[np.all(image_OD > self.optical_density_threshold, axis=1)]

#         # dictionary learning
#         # need to first update
#         # see https://github.com/dmlc/xgboost/issues/1715#issuecomment-420305786
#         os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
#         dictionary = spams.trainDL(
#             X=OD.T,
#             K=2,
#             lambda1=self.sparsity_regularizer,
#             mode=2,
#             modeD=0,
#             posAlpha=True,
#             posD=True,
#             verbose=False,
#         )
#         dictionary = normalize_matrix_cols(dictionary)
#         # order H and E.
#         # H on first col.
#         if dictionary[0, 0] > dictionary[1, 0]:
#             dictionary = dictionary[:, [1, 0]]
#         return dictionary

#     def _estimate_stain_vectors_macenko(self, image):
#         """
#         Estimate stain vectors using Macenko method. Returns a (3, 2) matrix with first column corresponding to
#         hematoxylin and second column corresponding to eosin in OD space.
#         Args:
#             image (np.ndarray): RGB image
#         """
#         # convert to Optical Density (OD) space
#         image_OD = RGB_to_OD(image)
#         # reshape to (M*N)x3
#         image_OD = image_OD.reshape(-1, 3)
#         # drop pixels with low OD
#         OD = image_OD[np.all(image_OD > self.optical_density_threshold, axis=1)]
#         # get top 2 PCs. PCs are eigenvectors of covariance matrix
#         try:
#             _, v = np.linalg.eigh(np.cov(OD.T))
#         except np.linalg.LinAlgError as err:
#             logger.exception(f"Error in computing eigenvectors: {err}")
#             raise
#         pcs = v[:, 1:3]
#         # project OD pixels onto plane of first 2 PCs
#         projected = OD @ pcs
#         # Calculate angle of each point on projection plane
#         angles = np.arctan2(projected[:, 1], projected[:, 0])
#         # get robust min and max angles
#         max_angle = np.percentile(angles, 100 * (1 - self.angular_percentile))
#         min_angle = np.percentile(angles, 100 * self.angular_percentile)
#         # get vector of unit length pointing in that angle, in projection plane
#         # unit length vector of angle theta is <cos(theta), sin(theta)>
#         v_max = np.array([np.cos(max_angle), np.sin(max_angle)])
#         v_min = np.array([np.cos(min_angle), np.sin(min_angle)])
#         # project back to OD space
#         stain1 = pcs @ v_max
#         stain2 = pcs @ v_min
#         # a heuristic to make the vector corresponding to hematoxylin first and the
#         # one corresponding to eosin second
#         if stain2[0] > stain1[0]:
#             HE = np.array((stain2, stain1)).T
#         else:
#             HE = np.array((stain1, stain2)).T
#         return HE

#     def _estimate_pixel_concentrations_lstsq(self, image, stain_matrix):
#         """
#         estimate concentrations of each stain at each pixel using least squares
#         Args:
#             image (np.ndarray): RGB image
#             stain_matrix (np.ndarray): matrix of H and E stain vectors in optical density (OD) space.
#                 Stain_matrix is (3, 2) and first column corresponds to hematoxylin by convention.
#         """
#         image_OD = RGB_to_OD(image).reshape(-1, 3)

#         # Get concentrations of each stain at each pixel
#         # image_ref.T = S @ C.T
#         #   image_ref.T is 3x(M*N)
#         #   stain matrix S is 3x2
#         #   concentration matrix C.T is 2x(M*N)
#         # solve for C using least squares
#         C = np.linalg.lstsq(stain_matrix, image_OD.T, rcond=None)[0].T
#         return C

#     def _estimate_pixel_concentrations_lasso(self, image, stain_matrix):
#         """
#         estimate concentrations of each stain at each pixel using lasso
#         Args:
#             image (np.ndarray): RGB image
#             stain_matrix (np.ndarray): matrix of H and E stain vectors in optical density (OD) space.
#                 Stain_matrix is (3, 2) and first column corresponds to hematoxylin by convention.
#         """
#         try:
#             import spams
#         except (ImportError, ModuleNotFoundError):
#             raise Exception(f"Vahadane method requires `spams` package to be installed")
#         image_OD = RGB_to_OD(image).reshape(-1, 3)

#         # Get concentrations of each stain at each pixel
#         # image_ref.T = S @ C.T
#         #   image_ref.T is 3x(M*N)
#         #   stain matrix S is 3x2
#         #   concentration matrix C.T is 2x(M*N)
#         # solve for C using lasso
#         lamb = self.regularizer_lasso
#         C = (
#             spams.lasso(X=image_OD.T, D=stain_matrix, mode=2, lambda1=lamb, pos=True)
#             .toarray()
#             .T
#         )
#         return C

#     def _reconstruct_image(self, pixel_intensities):
#         """
#         Reconstruct an image from pixel intensities. Uses reference stain matrix and max_c
#         from :func:`~pathml.preprocessing.transforms.StainNormalizationHE.fit_to_reference`, if that method has been
#         called, otherwise uses defaults.
#         Args:
#             pixel_intensities (np.ndarray): matrix of stain intensities for each pixel.
#             If image_ref is MxN, stain matrix is 2x(M*M)
#         """
#         # scale to max intensities
#         # actually use 99th percentile so it's more robust
#         max_c = np.percentile(pixel_intensities, 99, axis=0).reshape((1, 2))
#         pixel_intensities *= self.max_c_target / max_c

#         if self.target == "normalize":
#             im = np.exp(-self.stain_matrix_target_od @ pixel_intensities.T)
#         elif self.target == "hematoxylin":
#             im = np.exp(
#                 -self.stain_matrix_target_od[:, 0].reshape(-1, 1)
#                 @ pixel_intensities[:, 0].reshape(-1, 1).T
#             )
#         elif self.target == "eosin":
#             im = np.exp(
#                 -self.stain_matrix_target_od[:, 1].reshape(-1, 1)
#                 @ pixel_intensities[:, 1].reshape(-1, 1).T
#             )
#         else:
#             raise Exception(
#                 f"Error: input target {self.target} is invalid. Must be one of 'normalize', 'eosin', 'hematoxylin'"
#             )

#         im = im * self.background_intensity
#         im = np.clip(im, a_min=0, a_max=255)
#         im = im.T.astype(np.uint8)
#         return im

#     def F(self, image:MetaArrayLike) -> MetaArrayLike:
#         # first estimate stain matrix for reference image_ref
#         stain_matrix = self._estimate_stain_vectors(image=image)

#         # next get pixel concentrations for reference image_ref
#         C = self._estimate_pixel_concentrations(image=image, stain_matrix=stain_matrix)

#         # next reconstruct the image_ref
#         im_reconstructed = self._reconstruct_image(pixel_intensities=C)

#         im_reconstructed = im_reconstructed.reshape(image.shape)
#         return im_reconstructed

#     def apply(self, tile):
#         assert isinstance(
#             image, BioImIm
#         ), f"image is type {type(image)} but must be BioImIm"
#         assert (
#             tile.slide_type.stain == "HE"
#         ), f"Tile has slide_type.stain={tile.slide_type.stain}, but must be 'HE'"
#         tile.image = self.F(tile.image)


class NucleusDetectionHE(BioImTransform):
    """
    Simple nucleus detection algorithm for H&E stained images.
    Works by first separating hematoxylin channel, then doing interpolation using superpixels,
    and finally using Otsu's method for binary thresholding.
    Args:
        stain_estimation_method (str): Method for estimating stain matrix. Defaults to "vahadane"
        superpixel_region_size (int): region_size parameter used for superpixel creation. Defaults to 10.
        n_iter (int): Number of iterations to run SLIC superpixel algorithm. Defaults to 30.
        mask_name (str): Name of mask that is created.
        stain_kwargs (dict): other arguments passed to ``StainNormalizationHE()``
    References:
        Hu, B., Tang, Y., Eric, I., Chang, C., Fan, Y., Lai, M. and Xu, Y., 2018. Unsupervised learning for cell-level
        visual representation in histopathology images with generative adversarial networks. IEEE journal of
        biomedical and health informatics, 23(3), pp.1316-1328.
    """

    def __init__(
        self,
        mask_name=None,
        stain_estimation_method="vahadane",
        superpixel_region_size=10,
        n_iter=30,
        **stain_kwargs,
    ):
        self.stain_estimation_method = stain_estimation_method
        self.superpixel_region_size = superpixel_region_size
        self.n_iter = n_iter
        self.stain_kwargs = stain_kwargs
        self.mask_name = mask_name

    def __repr__(self):
        return (
            f"NucleusDetectionHE(mask_name={self.mask_name}, "
            f"stain_estimation_method={self.stain_estimation_method}, "
            f"superpixel_region_size={self.superpixel_region_size}, n_iter={self.n_iter}, "
            f"stain_kwargs={self.stain_kwargs})"
        )

    def F(self, image: MetaArrayLike) -> MetaArrayLike:
        assert image.dtype == np.uint8, f"Input image dtype {image.dtype} must be np.uint8"
        # im_hematoxylin = StainNormalizationHE(
        #     target="hematoxylin",
        #     stain_estimation_method=self.stain_estimation_method,
        #     **self.stain_kwargs,
        # ).F(image)
        im_interpolated = SuperpixelInterpolation(region_size=self.superpixel_region_size, n_iter=self.n_iter).F(image)
        # im_interp_grey = RGB_to_GREY(im_interpolated)
        thresholded = BinaryThreshold(use_otsu=True).F(im_interpolated)
        # flip sign so that nuclei regions are TRUE (255)
        thresholded = ~thresholded
        return thresholded

    def apply(self, image):
        # assert isinstance(image, BioImIm), f"image is type {type(image)} but must be BioImIm"
        assert self.mask_name is not None, "mask_name is None. Must supply a valid mask name"
        # assert (
        #     tile.slide_type.stain == "HE"
        # ), f"Tile has slide_type.stain={tile.slide_type.stain}, but must be 'HE'"
        nucleus_mask = self.F(image.image)
        image.mask = nucleus_mask

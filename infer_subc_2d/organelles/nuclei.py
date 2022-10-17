from scipy.ndimage import median_filter, gaussian_filter

from skimage.measure import label

from aicssegmentation.core.pre_processing_utils import intensity_normalization
from skimage.morphology import remove_small_holes  # function for post-processing (size filter)

from infer_subc_2d.utils.img import threshold_li_log, size_filter_2D

# from .qc import ObjectCheck, ObjectStats, ArrayLike


# def NucleiCheck(ObjectCheck):
#     """
#     Checker class for NUCLEI priors
#     """
#     def __init__(self, priors: ObjectStats)
#         self.prior = priors

#     @property
#     def self.prior(self):
#         if self.__stats is None:
#             return
#         return self.__stats

#     @prior.setter
#     def self.prior(self, prior: ObjectStats):
#         return self.__stats

#     def self.check_prior(self, test_image:ArrayLike):
#         pass


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
        holds the needed parameters (though they are not used)

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
    out_p = in_params.copy()

    ###################
    # PRE_PROCESSING
    ###################
    # TODO: replace params below with the input params
    scaling_param = [0]
    struct_img = intensity_normalization(struct_img, scaling_param=scaling_param)
    out_p["intensity_norm_param"] = scaling_param

    med_filter_size = 4
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    struct_img = median_filter(struct_img, size=med_filter_size)
    out_p["median_filter_size"] = med_filter_size

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    struct_img = gaussian_filter(
        struct_img, sigma=gaussian_smoothing_sigma, mode="nearest", truncate=gaussian_smoothing_truncate_range
    )
    out_p["gaussian_smoothing_sigma"] = gaussian_smoothing_sigma
    out_p["gaussian_smoothing_truncate_range"] = gaussian_smoothing_truncate_range

    ###################
    # CORE_PROCESSING
    ###################
    # struct_obj = struct_img > filters.threshold_li(struct_img)
    threshold_value_log = threshold_li_log(struct_img)

    threshold_factor = 0.9  # from cellProfiler
    thresh_min = 0.1
    thresh_max = 1.0
    threshold = min(max(threshold_value_log * threshold_factor, thresh_min), thresh_max)
    out_p["threshold_factor"] = threshold_factor
    out_p["thresh_min"] = thresh_min
    out_p["thresh_max"] = thresh_max

    struct_obj = struct_img > threshold

    ###################
    # POST_PROCESSING
    ###################
    hole_width = 5
    # # wrapper to remoce_small_objects
    struct_obj = remove_small_holes(struct_obj, hole_width**2)
    out_p["hole_width"] = hole_width

    small_object_max = 5
    struct_obj = size_filter_2D(struct_obj, min_size=small_object_max**2, connectivity=1)
    out_p["small_object_max"] = small_object_max

    retval = (struct_obj, label(struct_obj), out_p)
    return retval

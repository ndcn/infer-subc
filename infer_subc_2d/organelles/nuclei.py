from skimage.measure import label

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.utils import hole_filling

from infer_subc_2d.utils.img import (
    size_filter_2D,
    min_max_intensity_normalization,
    median_filter_slice_by_slice,
    apply_log_li_threshold,
    apply_mask,
)
from infer_subc_2d.constants import (
    TEST_IMG_N,
    NUC_CH,
    LYSO_CH,
    MITO_CH,
    GOLGI_CH,
    PEROXI_CH,
    ER_CH,
    LIPID_CH,
    RESIDUAL_CH,
)

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


def infer_NUCLEI(in_img, soma_mask) -> tuple:
    """
    Procedure to infer NUCLEI from linearly unmixed input.

    Parameters:
    ------------
    in_img: np.ndarray
        a 3d image containing all the channels

    soma_mask: np.ndarray
        mask

    Returns:
    -------------
    nuclei_object
        mask defined extent of NU

    """

    ###################
    # PRE_PROCESSING
    ###################
    nuclei = min_max_intensity_normalization(in_img[NUC_CH].copy())

    med_filter_size = 4
    # structure_img_median_3D = ndi.median_filter(struct_img,    size=med_filter_size  )
    nuclei = median_filter_slice_by_slice(nuclei, size=med_filter_size)

    gaussian_smoothing_sigma = 1.34
    gaussian_smoothing_truncate_range = 3.0
    nuclei = image_smoothing_gaussian_slice_by_slice(
        nuclei, sigma=gaussian_smoothing_sigma, truncate_range=gaussian_smoothing_truncate_range
    )

    ###################
    # CORE_PROCESSING
    ###################
    # struct_obj = struct_img > filters.threshold_li(struct_img)
    threshold_factor = 0.9  # from cellProfiler
    thresh_min = 0.1
    thresh_max = 1.0
    NU_object = apply_log_li_threshold(
        nuclei, threshold_factor=threshold_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )

    NU_labels = label(NU_object)
    ###################
    # POST_PROCESSING
    ###################
    hole_width = 5
    # # wrapper to remoce_small_objects
    NU_object = hole_filling(NU_object, hole_min=0, hole_max=hole_width**2, fill_2d=True)

    small_object_max = 45
    NU_object = size_filter_2D(NU_object, min_size=small_object_max**2, connectivity=1)

    return apply_mask(NU_object, soma_mask)

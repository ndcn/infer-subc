from skimage.measure import label
import numpy as np
from typing import Optional

from aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.utils import hole_filling

from infer_subc_2d.utils.img import (
    size_filter_2D,
    min_max_intensity_normalization,
    median_filter_slice_by_slice,
    apply_log_li_threshold,
    apply_mask,
    select_channel_from_raw,
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

# copy this to base.py for easy import

##########################
#  _infer_nuclei
##########################
def infer_nuclei(
    in_img: np.ndarray,
    soma_mask: np.ndarray,
    median_sz: int,
    gauss_sig: float,
    thresh_factor: float,
    thresh_min: float,
    thresh_max: float,
    max_hole_w: int,
    small_obj_w: int,
) -> np.ndarray:

    """
    Procedure to infer nuclei from linearly unmixed input,
    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    soma_mask:
        mask of soma extent
    median_sz:
        width of median filter for signal
    gauss_sig:
        sigma for gaussian smoothing of  signal
    thresh_factor:
        adjustment factor for log Li threholding
    thresh_min:
        abs min threhold for log Li threholding
    thresh_max:
        abs max threhold for log Li threholding
    max_hole_w:
        hole filling cutoff for nuclei post-processing
    small_obj_w:
        minimu object size cutoff for nuclei post-processing

    Returns
    -------------
    nuclei_object
        mask defined extent of NU

    """
    nuc_ch = NUC_CH
    nuclei = select_channel_from_raw(in_img, nuc_ch)

    ###################
    # PRE_PROCESSING
    ###################
    nuclei = min_max_intensity_normalization(nuclei)

    nuclei = median_filter_slice_by_slice(nuclei, size=median_sz)

    nuclei = image_smoothing_gaussian_slice_by_slice(nuclei, sigma=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    nuclei_object = apply_log_li_threshold(
        nuclei, thresh_factor=thresh_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )
    # NU_labels = label(nuclei_object)

    ###################
    # POST_PROCESSING
    ###################
    nuclei_object = hole_filling(nuclei_object, hole_min=0, hole_max=max_hole_w**2, fill_2d=True)

    if soma_mask is not None:
        nuclei_object = apply_mask(nuclei_object, soma_mask)

    nuclei_object = size_filter_2D(nuclei_object, min_size=small_obj_w**2, connectivity=1)

    return nuclei_object


##########################
#  fixed_infer_nuclei
##########################
def fixed_infer_nuclei(in_img: np.ndarray, soma_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Procedure to infer soma from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels
    soma_mask: Optional[np.ndarray] = None
        mask of soma extent

    Returns
    -------------
    nuclei_object
        mask defined extent of NU

    """
    nuc_ch = NUC_CH
    median_sz = 4
    gauss_sig = 1.34
    thresh_factor = 0.9
    thresh_min = 0.1
    thresh_max = 1.0
    max_hole_w = 5
    small_obj_w = 15

    return infer_nuclei(
        in_img, soma_mask, median_sz, gauss_sig, thresh_factor, thresh_min, thresh_max, max_hole_w, small_obj_w
    )

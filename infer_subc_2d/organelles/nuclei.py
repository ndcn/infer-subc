import numpy as np
from typing import Union

from infer_subc_2d.utils.img import (
    fill_and_filter_linear_size,
    apply_log_li_threshold,
    select_channel_from_raw,
    scale_and_smooth,
)
from infer_subc_2d.constants import NUC_CH

##########################
#  infer_nuclei
##########################
def infer_nuclei(
    in_img: np.ndarray,
    nuc_ch: Union[int, None],
    median_sz: int,
    gauss_sig: float,
    thresh_factor: float,
    thresh_min: float,
    thresh_max: float,
    max_hole_w: int,
    small_obj_w: int,
) -> np.ndarray:
    """
    Procedure to infer nuclei from linearly unmixed input.

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels; np.ndarray
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

    ###################
    # PRE_PROCESSING
    ###################
    if nuc_ch is None:
        nuc_ch = NUC_CH

    nuclei = select_channel_from_raw(in_img, nuc_ch)

    nuclei = scale_and_smooth(nuclei, median_sz=median_sz, gauss_sig=gauss_sig)

    ###################
    # CORE_PROCESSING
    ###################
    nuclei_object = apply_log_li_threshold(
        nuclei, thresh_factor=thresh_factor, thresh_min=thresh_min, thresh_max=thresh_max
    )

    ###################
    # POST_PROCESSING
    ###################
    nuclei_object = fill_and_filter_linear_size(nuclei_object, hole_min=0, hole_max=max_hole_w, min_size=small_obj_w)
    return nuclei_object


##########################
#  fixed_infer_nuclei
##########################
def fixed_infer_nuclei(in_img: np.ndarray) -> np.ndarray:
    """
    Procedure to infer soma from linearly unmixed input, with a *fixed* set of parameters for each step in the procedure.  i.e. "hard coded"

    Parameters
    ------------
    in_img:
        a 3d image containing all the channels

    Returns
    -------------
    nuclei_object
        inferred nuclei

    """
    nuc_ch = NUC_CH
    median_sz = 4
    gauss_sig = 1.34
    thresh_factor = 0.9
    thresh_min = 0.1
    thresh_max = 1.0
    max_hole_w = 25
    small_obj_w = 15

    return infer_nuclei(
        in_img, nuc_ch, median_sz, gauss_sig, thresh_factor, thresh_min, thresh_max, max_hole_w, small_obj_w
    )

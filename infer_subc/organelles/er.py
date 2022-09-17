from scipy.ndimage import median_filter, extrema
from scipy.interpolate import RectBivariateSpline

from skimage import img_as_float, filters
from skimage import morphology
from skimage.morphology import remove_small_objects, ball, disk, dilation, binary_closing, white_tophat, black_tophat
from skimage.filters import threshold_triangle, threshold_otsu, threshold_li, threshold_multiotsu, threshold_sauvola
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# from napari_aicsimageio.core import  reader_function
import aicssegmentation
from aicssegmentation.core.seg_dot import dot_3d_wrapper, dot_slice_by_slice, dot_2d_slice_by_slice_wrapper, dot_3d

from aicssegmentation.core.utils import topology_preserving_thinning, hole_filling, size_filter
from aicssegmentation.core.MO_threshold import MO
from aicssegmentation.core.vessel import filament_2d_wrapper, vesselnessSliceBySlice
from aicssegmentation.core.output_utils import save_segmentation, generate_segmentation_contour
from aicssegmentation.core.pre_processing_utils import (
    intensity_normalization,
    image_smoothing_gaussian_3d,
    image_smoothing_gaussian_slice_by_slice,
    edge_preserving_smoothing_3d,
)


from infer_subc.utils.img import *

##########################
#  infer_ENDOPLASMIC_RETICULUM
##########################
def infer_ENDOPLASMIC_RETICULUM(struct_img, CY_object, in_params) -> tuple:
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
    out_p = in_params.copy()

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
    struct_obj = remove_small_objects(struct_obj > 0, min_size=min_area, connectivity=1, in_place=False)
    out_p["min_area"] = min_area

    retval = (struct_obj, out_p)
    return retval

import numpy as np
from skimage.morphology import binary_erosion, binary_dilation


##########################
#  infer_cytosol
##########################
def infer_cytosol(nuclei_object: np.ndarray, soma_mask: np.ndarray, erode_nuclei: bool = True) -> np.ndarray:
    """
    Procedure to infer infer from linearly unmixed input. (logical soma AND NOT nucleus)

    Parameters
    ------------
    nuclei_object: np.ndarray
        a 3d image containing the nuclei object

    soma_mask: np.ndarray
        a 3d image containing the soma object (mask)

    erode_nuclei: bool
        should we erode?

    Returns
    -------------
    cytosol_mask: np.ndarray (bool)

    """

    if erode_nuclei:
        cytosol_mask = np.logical_xor(soma_mask, binary_erosion(nuclei_object))
    else:
        cytosol_mask = np.logical_xor(soma_mask, nuclei_object)

    return cytosol_mask

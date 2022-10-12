import numpy as np
from skimage.morphology import binary_erosion, binary_dilation


##########################
#  infer_CYTOSOL
##########################
def infer_CYTOSOL(SO_object, NU_object, erode_NU=True, dilate=True):
    """
    Procedure to infer CYTOSOL from linearly unmixed input.

    Parameters:
    ------------
    SO_object: np.ndarray
        a 3d image containing the NUCLEI signal

    NU_object: np.ndarray
        a 3d image containing the NUCLEI signal

    erode_NU: bool
        should we erode?

    Returns:
    -------------
    CY_object: np.ndarray (bool)

    """

    # NU_eroded1 = morphology.binary_erosion(NU_object,  footprint=morphology.ball(3) )
    if erode_NU:
        CY_object = np.logical_and(SO_object, ~binary_erosion(NU_object))
    else:
        CY_object = np.logical_and(SO_object, ~NU_object)

    if dilate:
        return binary_dilation(CY_object)
    else:
        return CY_object

import numpy as np
from typing import Dict
from pathlib import Path
import time

from skimage.morphology import binary_erosion
from infer_subc.core.file_io import export_inferred_organelle, import_inferred_organelle
from infer_subc.core.img import apply_mask, label_bool_as_uint16


##########################
#  infer_cytoplasm
##########################
def infer_cytoplasm(nuclei_object: np.ndarray, cellmask: np.ndarray, erode_nuclei: bool = True) -> np.ndarray:
    """
    Procedure to infer infer from linearly unmixed input. (logical cellmask AND NOT nucleus)

    Parameters
    ------------
    nuclei_object:
        a 3d image containing the nuclei object
    cellmask:
        a 3d image containing the cellmask object (mask)
    erode_nuclei:
        should we erode?

    Returns
    -------------
    cytoplasm_mask
        boolean np.ndarray

    """
    nucleus_obj = apply_mask(nuclei_object, cellmask)

    if erode_nuclei:
        cytoplasm_mask = np.logical_xor(cellmask, binary_erosion(nucleus_obj))
    else:
        cytoplasm_mask = np.logical_xor(cellmask, nucleus_obj)

    return label_bool_as_uint16(cytoplasm_mask)


def infer_and_export_cytoplasm(
    nuclei_object: np.ndarray, cellmask: np.ndarray, meta_dict: Dict, out_data_path: Path
) -> np.ndarray:
    """
    infer nucleus and write inferred nuclei to ome.tif file

    Parameters
    ------------
    nuclei_object:
        a 3d image containing the nuclei object
    cellmask:
        a 3d image containing the cellmask object (mask)
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """
    cytoplasm = infer_cytoplasm(nuclei_object, cellmask)

    out_file_n = export_inferred_organelle(cytoplasm, "cyto", meta_dict, out_data_path)
    print(f"inferred cytoplasm. wrote {out_file_n}")
    return cytoplasm


def get_cytoplasm(nuclei_obj: np.ndarray, cellmask: np.ndarray, meta_dict: Dict, out_data_path: Path) -> np.ndarray:
    """
    load cytoplasm if it exists, otherwise calculate and write to ome.tif file

    Parameters
    ------------
    in_img:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    cellmask:
        a 3d image containing the cellmask object (mask)
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """
    try:
        cytoplasm = import_inferred_organelle("cyto", meta_dict, out_data_path)>0
    except:
        start = time.time()
        print("starting segmentation...")
        cytoplasm = infer_and_export_cytoplasm(nuclei_obj, cellmask, meta_dict, out_data_path)
        end = time.time()
        print(f"inferred cytoplasm in ({(end - start):0.2f}) sec")

    return cytoplasm

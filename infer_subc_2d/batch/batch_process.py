from typing import Union
from pathlib import Path
import numpy as np

from infer_subc_2d.utils.file_io import export_infer_organelles, read_czi_image, list_image_files
from infer_subc_2d.utils.img import select_z_from_raw

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

from infer_subc_2d.organelles import (
    fixed_infer_soma,
    fixed_infer_nuclei,
    infer_cytosol,
    find_optimal_Z,
    fixed_get_optimal_Z_image,
    fixed_find_optimal_Z,
    fixed_infer_lysosomes,
    fixed_infer_mitochondria,
    fixed_infer_golgi,
    fixed_infer_endoplasmic_reticulum,
    fixed_infer_peroxisome,
    fixed_infer_lipid,
)

###########
# infer organelles
##########
def fixed_infer_organelles(img_data):
    """
    wrapper to infer all organelles from a single multi-channel image
    """
    # ch_to_agg = (LYSO_CH, MITO_CH, GOLGI_CH, PEROXI_CH, ER_CH, LIPID_CH)

    # nuc_ch = NUC_CH
    # optimal_Z = find_optimal_Z(img_data, nuc_ch, ch_to_agg)
    # # Stage 1:  nuclei, soma, cytosol
    # img_2D = select_z_from_raw(img_data, optimal_Z)
    img_2D = fixed_get_optimal_Z_image(img_data)

    soma_mask = fixed_infer_soma(img_2D)

    nuclei_object = fixed_infer_nuclei(img_2D, soma_mask)

    cytosol_mask = infer_cytosol(nuclei_object, soma_mask)

    # cyto masked objects.
    lysosome_object = fixed_infer_lysosomes(img_2D, cytosol_mask)
    mito_object = fixed_infer_mitochondria(img_2D, cytosol_mask)
    golgi_object = fixed_infer_golgi(img_2D, cytosol_mask)
    peroxi_object = fixed_infer_peroxisome(img_2D, cytosol_mask)
    er_object = fixed_infer_endoplasmic_reticulum(img_2D, cytosol_mask)
    lipid_object = fixed_infer_lipid(img_2D, cytosol_mask)

    img_layers = [
        nuclei_object,
        lysosome_object,
        mito_object,
        golgi_object,
        peroxi_object,
        er_object,
        lipid_object,
        soma_mask,
        cytosol_mask,
    ]

    layer_names = [
        "nuclei",
        "lysosome",
        "mitochondria",
        "golgi",
        "peroxisome",
        "er",
        "lipid_body",
        "soma_mask",
        "cytosol_mask",
    ]
    # TODO: pack outputs into something napari readable
    img_out = np.stack(img_layers, axis=0)
    return (img_out, layer_names, optimal_Z)


def batch_process_all_czi(data_root_path, source_dir: Union[Path, str] = "raw"):
    """ """
    # linearly unmixed ".czi" files are here
    data_path = data_root_path / "raw"
    im_type = ".czi"
    # get the list of all files
    img_file_list = list_image_files(data_path, im_type)
    files_generated = []
    for czi_file in img_file_list:
        out_fn = process_czi_image(czi_file)
        files_generated.append(out_fn)

    print(f"generated {len(files_generated)} ")
    return files_generated


def process_czi_image(czi_file_name):
    """wrapper for processing"""

    img_data, meta_dict = read_czi_image(czi_file_name)
    # # get some top-level info about the RAW data
    # channel_names = meta_dict['name']
    # img = meta_dict['metadata']['aicsimage']
    # scale = meta_dict['scale']
    # channel_axis = meta_dict['channel_axis']

    inferred_organelles, layer_names, optimal_Z = fixed_infer_organelles(img_data)
    meta_dict["z_slice"] = optimal_Z
    out_file_n = export_infer_organelles(inferred_organelles, layer_names, meta_dict, data_root_path)

    ## TODO:  collect stats...

    return out_file_n


def stack_organelle_objects(
    soma_mask,
    nuclei_object,
    cytosol_mask,
    lysosome_object,
    mito_object,
    golgi_object,
    peroxi_object,
    er_object,
    lipid_object,
) -> np.ndarray:
    """wrapper to stack the inferred objects into a single numpy.ndimage"""
    img_layers = [
        soma_mask,
        nuclei_object,
        cytosol_mask,
        lysosome_object,
        mito_object,
        golgi_object,
        peroxi_object,
        er_object,
        lipid_object,
    ]
    return np.stack(img_layers, axis=0)


def stack_organelle_layers(*layers) -> np.ndarray:
    """wrapper to stack the inferred objects into a single numpy.ndimage"""

    return np.stack(layers, axis=0)

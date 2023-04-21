from typing import Union
from pathlib import Path
import numpy as np

from infer_subc.core.file_io import export_infer_organelles, read_czi_image, list_image_files
from infer_subc.core.img import select_z_from_raw


from infer_subc.constants import (
    TEST_IMG_N,
    NUC_CH,
    LYSO_CH,
    MITO_CH,
    GOLGI_CH,
    PEROX_CH,
    ER_CH,
    LD_CH,
    RESIDUAL_CH,
)

from infer_subc.organelles import (
    fixed_infer_cellmask_fromaggr,
    fixed_infer_nuclei,
    infer_cytoplasm,
    fixed_infer_lyso,
    fixed_infer_mito,
    fixed_infer_golgi,
    fixed_infer_ER,
    fixed_infer_perox,
    fixed_infer_LD,
)


###########
# infer organelles
##########
def fixed_infer_organelles(img_data):
    """
    wrapper to infer all organelles from a single multi-channel image
    """
    # ch_to_agg = (LYSO_CH, MITO_CH, GOLGI_CH, PEROX_CH, ER_CH, LD_CH)

    # nuc_ch = NUC_CH
    # optimal_Z = find_optimal_Z(img_data, nuc_ch, ch_to_agg)
    # optimal_Z = fixed_find_optimal_Z(img_data)
    # # Stage 1:  nuclei, cellmask, cytoplasm
    # img_2D = fixed_get_optimal_Z_image(img_data)

    cellmask = fixed_infer_cellmask_fromaggr(img_data)

    nuclei_object = fixed_infer_nuclei(img_data, cellmask)

    cytoplasm_mask = infer_cytoplasm(nuclei_object, cellmask)

    # cyto masked objects.
    lyso_object = fixed_infer_lyso(img_data, cytoplasm_mask)
    mito_object = fixed_infer_mito(img_data, cytoplasm_mask)
    golgi_object = fixed_infer_golgi(img_data, cytoplasm_mask)
    peroxi_object = fixed_infer_perox(img_data, cytoplasm_mask)
    er_object = fixed_infer_ER(img_data, cytoplasm_mask)
    LD_object = fixed_infer_LD(img_data, cytoplasm_mask)

    img_layers = [
        nuclei_object,
        lyso_object,
        mito_object,
        golgi_object,
        peroxi_object,
        er_object,
        LD_object,
        cellmask,
        cytoplasm_mask,
    ]

    layer_names = [
        "nuclei",
        "lyso",
        "mitochondria",
        "golgi",
        "peroxisome",
        "er",
        "LD_body",
        "cellmask",
        "cytoplasm_mask",
    ]
    # TODO: pack outputs into something napari readable
    img_out = np.stack(img_layers, axis=0)
    return (img_out, layer_names)


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


def process_czi_image(czi_file_name, data_root_path):
    """wrapper for processing"""

    img_data, meta_dict = read_czi_image(czi_file_name)
    # # get some top-level info about the RAW data
    # channel_names = meta_dict['name']
    # img = meta_dict['metadata']['aicsimage']
    # scale = meta_dict['scale']
    # channel_axis = meta_dict['channel_axis']

    inferred_organelles, layer_names, optimal_Z = fixed_infer_organelles(img_data)
    out_file_n = export_infer_organelles(inferred_organelles, layer_names, meta_dict, data_root_path)

    ## TODO:  collect stats...

    return out_file_n


def stack_organelle_objects(
    cellmask,
    nuclei_object,
    cytoplasm_mask,
    lyso_object,
    mito_object,
    golgi_object,
    peroxi_object,
    er_object,
    LD_object,
) -> np.ndarray:
    """wrapper to stack the inferred objects into a single numpy.ndimage"""
    img_layers = [
        cellmask,
        nuclei_object,
        cytoplasm_mask,
        lyso_object,
        mito_object,
        golgi_object,
        peroxi_object,
        er_object,
        LD_object,
    ]
    return np.stack(img_layers, axis=0)


def stack_organelle_layers(*layers) -> np.ndarray:
    """wrapper to stack the inferred objects into a single numpy.ndimage"""

    return np.stack(layers, axis=0)

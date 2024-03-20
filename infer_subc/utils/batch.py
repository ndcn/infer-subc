from typing import Union, Optional, Dict, List

from pathlib import Path
import numpy as np

from infer_subc.core.file_io import export_inferred_organelle, read_czi_image, list_image_files, export_tiff, read_tiff_image


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


from infer_subc.core.img import label_uint16




from infer_subc.organelles import (
    fixed_infer_cellmask_fromcomposite,
    fixed_infer_nuclei_fromlabel,
    infer_cytoplasm,
    fixed_infer_lyso,
    fixed_infer_mito,
    fixed_infer_golgi,
    fixed_infer_ER,
    fixed_infer_perox,
    fixed_infer_LD,
)


def explode_mask(mask_path: Union[Path,str], postfix: str= "masks", im_type: str = ".tiff") -> bool:
    """ 
    Split a 3 channel 'masks' file into 'nuc', 'cell', and 'cyto' images.  WARNING: requires the channels to be nuc = 0, cell = 1, and cyto = 2
    TODO: add logging instead of printing
        append tiffcomments with provenance
    """
    if isinstance(mask_path, str): mask_path = Path(mask_path)
    # load image 
    full_stem = mask_path.stem
    if full_stem.endswith(postfix):
        stem = full_stem.rstrip(postfix)
        image = read_tiff_image(mask_path)
        assert image.shape[0]==3
        
        # make into np.uint16 labels
        nuclei = label_uint16(image[0])
        # export as np.uint8 (255)
        cellmask = image[1]>0            
        cytoplasm = image[2]>0

        # write wasks
        root_stem = mask_path.parent / stem
        # ret1 = imwrite(f"{root}nuclei{stem}", nuclei)
        ret1 = export_tiff(nuclei, f"{stem}nuc", mask_path.parent, None)
        # ret2 = imwrite(f"{root}cellmask{stem}", cellmask)
        ret2 = export_tiff(cellmask, f"{stem}cell", mask_path.parent, None)
        # ret3 = imwrite(f"{root}cytosol{stem}", cytosol)
        ret3 = export_tiff(cytoplasm, f"{stem}cyto", mask_path.parent, None)

        # print(f"wrote {stem}-{{nuclei,cellmask,cyto}}")
        return True
    else:
        return False
    

def explode_masks(root_path: Union[Path,str], postfix: str= "masks", im_type: str = ".tiff"):
    """  
    TODO: add loggin instead of printing
        append tiffcomments with provenance
    """
    if isinstance(root_path, str): root_path = Path(root_path)

    img_file_list = list_image_files(root_path,im_type, postfix)
    wrote_cnt = 0
    for img_f in img_file_list:
        if explode_mask(img_f, postfix=postfix, im_type=im_type): 
            wrote_cnt += 1
        else: 
            print(f"failed to explode {img_f}")
    # else:
    #     print(f"how thefark!!! {img_f}")
   

    print(f"exploded {wrote_cnt*100./len(img_file_list)} pct of {len(img_file_list)} files")
    return wrote_cnt


def find_segmentation_tiff_files(prototype:Union[Path,str],
                                  name_list:List[str], 
                                  seg_path:Union[Path,str],
                                  suffix:Union[str, None]=None) -> Dict:
    """
    Find the matching segmentation files to the raw image file based on the raw image file path.

    Paramters:
    ---------
    prototype:Union[Path,str]
        the file path (as a string) for one raw image file; this file should have matching segmentation 
        output files with the same file name root and different file name ending that match the strings 
        provided in name_list
    name_list:List[str]
        a list of file name endings related to what segmentation is that file
    seg_path:Union[Path,str]
        the path (as a string) to the matching segmentation files.
    suffix:Union[str, None]=None
        any additional text that exists between the file root and the name_list ending
        Ex) Prototype = "C:/Users/Shannon/Documents/Python_Scripts/Infer-subc/raw/a48hrs-Ctrl_9_Unmixing.czi"
            Name of organelle file = a48hrs-Ctrl_9_Unmixing-20230426_test_cell.tiff
            result of .stem = "a48hrs-Ctrl_9_Unmixing"
            organelle/cell area type = "cell"
            suffix = "-20230426_test_"
    
    Returns:
    ----------
    a dictionary of file paths for each image type (raw and all the different segmentations)

    """
    # raw
    prototype = Path(prototype)
    if not prototype.exists():
        print(f"bad prototype. please choose an existing `raw` file as prototype")
        return dict()

    out_files = {"raw":prototype}
    seg_path = Path(seg_path) 

    # raw
    if not seg_path.is_dir():
        print(f"bad path argument. please choose an existing path containing organelle segmentations")
        return out_files

    # segmentations
    for org_n in name_list:
        org_name = Path(seg_path) / f"{prototype.stem}{suffix}{org_n}.tiff"
        if org_name.exists(): 
            out_files[org_n] = org_name
        else: 
            print(f"{org_n} .tiff file not found in {seg_path} returning")
            out_files[org_n] = None
    
    return out_files 



# def find_segmentation_tiff_files(prototype:Union[Path,str], organelles: List[str], int_path: Union[Path,str]) -> Dict:
#     """
#     find the nescessary image files based on protype, the organelles involved, and paths
#     """

#     # raw
#     prototype = Path(prototype)
#     if not prototype.exists():
#         print(f"bad prototype. please choose an existing `raw` file as prototype")
#         return dict()
#     # make sure protoype ends with czi

#     out_files = {"raw":prototype}

#     int_path = Path(int_path) 
#     # raw
#     if not int_path.is_dir():
#         print(f"bad path argument. please choose an existing path containing organelle segmentations")
#         return out_files
    
#     # cyto, cellmask
#     cyto_nm = int_path / f"{prototype.stem}-cyto.tiff"
#     if cyto_nm.exists():
#         out_files["cyto"] = cyto_nm
#     else:
#         print(f"cytosol mask not found.  We'll try to extract from masks ")
#         if explode_mask(int_path / f"{prototype.stem}-masks.tiff"): 
#             out_files["cyto"] = cyto_nm
#         else: 
#             print(f"failed to explode {prototype.stem}-masks.tiff")
#             return out_files
    
#     cellmask_nm = int_path / f"{prototype.stem}-cell.tiff"
#     if  cellmask_nm.exists():
#         out_files["cell"] = cellmask_nm
#     else:
#         print(f"cellmask file not found in {int_path} returning")
#         out_files["cell"] = None

#     # organelles
#     for org_n in organelles:
#         org_name = Path(int_path) / f"{prototype.stem}-{org_n}.tiff"
#         if org_name.exists(): 
#             out_files[org_n] = org_name
#         else: 
#             print(f"{org_n} .tiff file not found in {int_path} returning")
#             out_files[org_n] = None
    
#     if "nuc" not in organelles:
#         nuc_nm = int_path / f"{prototype.stem}-nuc.tiff"
#         if  nuc_nm.exists():
#             out_files["nuc"] = nuc_nm
#         else:
#             print(f"nuc file not found in {int_path} returning")
#             out_files["nuc"] = None



#     return out_files


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
        "cell",
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
    out_file_n = export_inferred_organelle(inferred_organelles, layer_names, meta_dict, data_root_path)

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

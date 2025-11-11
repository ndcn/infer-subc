from typing import Union, Dict, List
from pathlib import Path
import time

from infer_subc.core.file_io import list_image_files, export_tiff, read_tiff_image, read_czi_image, export_inferred_organelle
from infer_subc.core.img import label_uint16, apply_mask, min_max_intensity_normalization, label, size_filter_linear_size

from infer_subc.organelles.masks import infer_masks, infer_masks_A, infer_masks_B
from infer_subc.organelles.er import infer_ER
from infer_subc.organelles.golgi import infer_golgi
from infer_subc.organelles.lipid import infer_LD
from infer_subc.organelles.lysosome import infer_lyso
from infer_subc.organelles.mitochondria import infer_mito
from infer_subc.organelles.peroxisome import infer_perox



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

### USED ###
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
        elif org_name.exists() == False: 
            org_name = Path(seg_path) / f"{prototype.stem}{suffix}{org_n}.tif"
            out_files[org_n] = org_name
        else: 
            print(f"{org_n} .tiff file not found in {seg_path} returning")
            out_files[org_n] = None
    
    return out_files



def batch_process_segmentation(raw_path: Union[Path,str],
                               raw_file_type: str,
                               seg_path: Union[Path, str],
                               name_suffix: Union[str, None],
                               masks_settings: Union[List, None],
                               masks_A_settings: Union[List, None],
                               masks_B_settings: Union[List, None],
                               lyso_settings: Union[List, None],
                               mito_settings: Union[List, None],
                               golgi_settings: Union[List, None],
                               perox_settings: Union[List, None],
                               ER_settings: Union[List, None],
                               LD_settings: Union[List, None]):
    """
    This function batch processes the segmentation workflows for multiple organelles and masks across multiple images.

    Parameters:
    ----------
    raw_path: Union[Path,str]
        A string or a Path object of the path to your raw (e.g., intensity) images that will be the input for segmentation
    raw_file_type: str
        The raw file type (e.g., ".tiff" or ".czi")
    seg_path: Union[Path, str]
        A string or a Path object of the path where the segmentation outputs will be saved 
    name_suffix: str
        An optional string to include before the segmentation suffix at the end of the output file. 
        For example, if the name_suffix was "20240105", the segmentation file output from the 1.1_masks workflow would include:
        "{base-file-name}-20240105-masks"
    {}_settings: Union[List, None]
        For each workflow that you wish to include in the batch processing, 
        fill out the information in the associated settings list. 
        The necessary settings for each function are included below.

    For infer_masks:
    `masks_settings` = [nuc_ch: Union[int,None],
                    nuc_median_sz: int, 
                    nuc_gauss_sig: float,
                    nuc_thresh_factor: float,
                    nuc_thresh_min: float,
                    nuc_thresh_max: float,
                    nuc_min_hole_w: int,
                    nuc_max_hole_w: int,
                    nuc_small_obj_w: int,
                    nuc_fill_filter_method: str,
                    cell_weights: list[int],
                    cell_rescale: bool,
                    cell_median_sz: int,
                    cell_gauss_sig: float,
                    cell_mo_method: str,
                    cell_mo_adjust: float,
                    cell_mo_cutoff_size: int,
                    cell_min_hole_w: int,
                    cell_max_hole_w: int,
                    cell_small_obj_w: int,
                    cell_fill_filter_method: str,
                    cell_watershed_method: str,
                    cyto_erode_nuclei = True]
    
    For infer_masks_A:
    `masks_A_settings` = [cyto_weights: list[int],
                        cyto_rescale: bool,
                        cyto_median_sz: int,
                        cyto_gauss_sig: float,
                        cyto_mo_method: str,
                        cyto_mo_adjust: float,
                        cyto_mo_cutoff_size: int,
                        cyto_min_hole_w: int,
                        cyto_max_hole_w: int,
                        cyto_small_obj_w: int,
                        cyto_fill_filter_method: str,
                        nuc_min_hole_w: int,
                        nuc_max_hole_w: int,
                        nuc_fill_method: str,
                        nuc_small_obj_w: int,
                        nuc_fill_filter_method: str,
                        cell_min_hole_width: int,
                        cell_max_hole_width: int,
                        cell_small_obj_width: int,
                        cell_fill_filter_method: str]

    For infer_masks_B:
    - `masks_B_settings` = [cyto_weights: list[int],
                        cyto_rescale: bool,
                        cyto_median_sz: int,
                        cyto_gauss_sig: float,
                        cyto_mo_method: str,
                        cyto_mo_adjust: float,
                        cyto_mo_cutoff_size: int,
                        cyto_min_hole_w: int,
                        cyto_max_hole_w: int,
                        cyto_small_obj_w: int,
                        cyto_fill_filter_method: str,
                        max_nuclei_width: int,
                        nuc_small_obj_width: int,
                        cell_fillhole_max: int,
                        cyto_small_object_width2: int]


    For infer_lyso:
    - `lyso_settings` = [lyso_ch: int,
                    median_sz: int,
                    gauss_sig: float,
                    dot_scale_1: float,
                    dot_cut_1: float,
                    dot_scale_2: float,
                    dot_cut_2: float,
                    dot_scale_3: float,
                    dot_cut_3: float,
                    dot_method: str,
                    fil_scale_1: float,
                    fil_cut_1: float,
                    fil_scale_2: float, 
                    fil_cut_2: float, 
                    fil_scale_3: float, 
                    fil_cut_3: float,
                    fil_method: str,
                    min_hole_w: int,
                    max_hole_w: int,
                    small_obj_w: int,
                    fill_filter_method: str]

    For infer_mito:
    - `mito_settings` = [mito_ch: int,
                        median_sz: int,
                        gauss_sig: float,
                        dot_scale_1: float,
                        dot_cut_1: float,
                        dot_scale_2: float,
                        dot_cut_2: float,
                        dot_scale_3: float,
                        dot_cut_3: float,
                        dot_method: str,
                        fil_scale_1: float,
                        fil_cut_1: float,
                        fil_scale_2: float, 
                        fil_cut_2: float, 
                        fil_scale_3: float, 
                        fil_cut_3: float,
                        fil_method: str,
                        min_hole_w: int,
                        max_hole_w: int,
                        small_obj_w: int,
                        fill_filter_method: str]

    For infer_golgi:
    - `golgi_settings` = [golgi_ch: int,
                        median_sz: int,
                        gauss_sig: float,
                        mo_method: str,
                        mo_adjust: float,
                        mo_cutoff_size: int,
                        min_thickness: int,
                        thin_dist: int,
                        dot_scale_1: float,
                        dot_cut_1: float,
                        dot_scale_2: float,
                        dot_cut_2: float,
                        dot_scale_3: float,
                        dot_cut_3: float,
                        dot_method: str,
                        min_hole_w: int,
                        max_hole_w: int,
                        small_obj_w: int,
                        fill_filter_method: str]

    For infer_perox:
    - `perox_settings` = [perox_ch: int,
                        median_sz: int,
                        gauss_sig: float,
                        dot_scale_1: float,
                        dot_cut_1: float,
                        dot_scale_2: float,
                        dot_cut_2: float,
                        dot_scale_3: float,
                        dot_cut_3: float,
                        dot_method: str,
                        hole_min_width: int,
                        hole_max_width: int,
                        small_object_width: int,
                        fill_filter_method: str]

    For infer_ER:
    - `ER_settings` = [ER_ch: int,
                median_sz: int,
                gauss_sig: float,
                MO_thresh_method: str,
                MO_cutoff_size: float,
                MO_thresh_adj: float,
                fil_scale_1: float,
                fil_cut_1: float,
                fil_scale_2: float, 
                fil_cut_2: float, 
                fil_scale_3: float, 
                fil_cut_3: float,
                fil_method: str,
                min_hole_w: int,
                max_hole_w: int,
                small_obj_w: int,
                fill_filter_method: str]

    For infer_LD:
    - `LD_settings` = [LD_ch: str,
                    median_sz: int,
                    gauss_sig: float,
                    method: str,
                    thresh_factor: float,
                    thresh_min: float,
                    thresh_max: float,
                    min_hole_w: int,
                    max_hole_w: int,
                    small_obj_w: int,
                    fill_filter_method: str]
    


    Returns:
    ----------

    """
    start = time.time()
    count = 0

    if isinstance(raw_path, str): raw_path = Path(raw_path)
    if isinstance(seg_path, str): seg_path = Path(seg_path)

    if not Path.exists(seg_path):
        Path.mkdir(seg_path)
        print(f"The specified 'seg_path' was not found. Creating {seg_path}.")
    
    if not name_suffix:
        name_suffix=""

    # reading list of files from the raw path
    img_file_list = list_image_files(raw_path, raw_file_type)

    for img in img_file_list:
        count = count + 1
        print(f"Beginning segmentation of: {img}")
        seg_list = []
        mask = None

        # read in raw file and metadata
        img_data, meta_dict = read_czi_image(img)

        # run masks function
        if masks_settings:
            masks = infer_masks(img_data, *masks_settings)
            export_inferred_organelle(masks, name_suffix+"masks", meta_dict, seg_path)
            seg_list.append("masks")
            if mask is None:
                mask = masks
            else:
                print("multiple mask segmentations made for same image")
        
        # run masks_A function
        if masks_A_settings:
            masks_A =  infer_masks_A(img_data, *masks_A_settings)
            export_inferred_organelle(masks_A, name_suffix+"masks_A", meta_dict, seg_path)
            seg_list.append("masks_A")
            if mask is None:
                mask = masks_A
            else:
                print("multiple mask segmentations made for same image")
            
        # run masks_B function
        if masks_B_settings:
            masks_B = infer_masks_B(img_data, *masks_B_settings)
            export_inferred_organelle(masks_B, name_suffix+"masks_B", meta_dict, seg_path)
            seg_list.append("masks_B")
            if mask is None:
                mask = masks_B
            else:
                print("multiple mask segmentations made for same image")

        # # run masks_C function
        # if masks_C_settings:
        #     masks_C = infer_masks_C(img_data, *masks_C_settings)
        #     export_inferred_organelle(masks_C, name_suffix+"masks_C", meta_dict, seg_path)
        #     seg_list.append("masks_C")
        #     if mask is None:
        #         mask = masks_C
        #     else:
        #         print("multiple mask segmentations made for same image")
        
        # # run masks_D function
        # if masks_D_settings:
        #     masks_D = infer_masks_D(img_data, *masks_D_settings)
        #     export_inferred_organelle(masks_D, name_suffix+"masks_D", meta_dict, seg_path)
        #     seg_list.append("masks_D")
        #     if mask is None:
        #         mask = masks_D
        #     else:
        #         print("multiple mask segmentations made for same image")

        # run 1.2_infer_lysosomes function
        if lyso_settings:
            lyso_seg = infer_lyso(img_data, *lyso_settings)
            export_inferred_organelle(lyso_seg, name_suffix+"lyso", meta_dict, seg_path)  
            seg_list.append("lyso")          

        if mito_settings:
            mito_seg = infer_mito(img_data, *mito_settings)
            export_inferred_organelle(mito_seg, name_suffix+"mito", meta_dict, seg_path)  
            seg_list.append("mito")
            
        if golgi_settings:
            golgi_seg = infer_golgi(img_data, *golgi_settings)
            export_inferred_organelle(golgi_seg, name_suffix+"golgi", meta_dict, seg_path)  
            seg_list.append("golgi")

        if perox_settings:
            perox_seg = infer_perox(img_data, *perox_settings)
            export_inferred_organelle(perox_seg, name_suffix+"perox", meta_dict, seg_path)  
            seg_list.append("perox")
            
        if ER_settings:
            ER_seg = infer_ER(img_data, *ER_settings)
            export_inferred_organelle(ER_seg, name_suffix+"ER", meta_dict, seg_path)  
            seg_list.append("ER")
            
        if LD_settings:
            LD_seg = infer_LD(img_data, *LD_settings)
            export_inferred_organelle(LD_seg, name_suffix+"LD", meta_dict, seg_path)
            seg_list.append("LD")
        
        # if som_neu_settings:
        #     som_neu_seg = infer_soma_neurites(in_seg=mask, multichannel_input=True, chan=1, *som_neu_settings)
        #     export_inferred_organelle(som_neu_seg, name_suffix+"soma_neurites", meta_dict, seg_path)  
        #     seg_list.append("soma_neurites")

        # if som_neu_settings:
        #     som_neu_seg = infer_soma_neurites(in_seg=mask, multichannel_input=True, chan=0, method=som_neu_seg[0])
        #     export_inferred_organelle(som_neu_seg, name_suffix+"soma_neurites", meta_dict, seg_path)  
        #     seg_list.append("soma_neurites")

        end = time.time()
        print(f"Processing for {img} completed in {(end - start)/60} minutes.")

    return print(f"Batch processing complete: {count} images segmented in {(end-start)/60} minutes.")





#################################################################
########################## DEPRICATING ##########################
#################################################################

###########
# infer organelles
##########
# def fixed_infer_organelles(img_data):
#     """
#     wrapper to infer all organelles from a single multi-channel image
#     """
#     # ch_to_agg = (LYSO_CH, MITO_CH, GOLGI_CH, PEROX_CH, ER_CH, LD_CH)

#     # nuc_ch = NUC_CH
#     # optimal_Z = find_optimal_Z(img_data, nuc_ch, ch_to_agg)
#     # optimal_Z = fixed_find_optimal_Z(img_data)
#     # # Stage 1:  nuclei, cellmask, cytoplasm
#     # img_2D = fixed_get_optimal_Z_image(img_data)

#     cellmask = fixed_infer_cellmask_fromaggr(img_data)

#     nuclei_object = fixed_infer_nuclei(img_data, cellmask)

#     cytoplasm_mask = infer_cytoplasm(nuclei_object, cellmask)

#     # cyto masked objects.
#     lyso_object = fixed_infer_lyso(img_data, cytoplasm_mask)
#     mito_object = fixed_infer_mito(img_data, cytoplasm_mask)
#     golgi_object = fixed_infer_golgi(img_data, cytoplasm_mask)
#     peroxi_object = fixed_infer_perox(img_data, cytoplasm_mask)
#     er_object = fixed_infer_ER(img_data, cytoplasm_mask)
#     LD_object = fixed_infer_LD(img_data, cytoplasm_mask)

#     img_layers = [
#         nuclei_object,
#         lyso_object,
#         mito_object,
#         golgi_object,
#         peroxi_object,
#         er_object,
#         LD_object,
#         cellmask,
#         cytoplasm_mask,
#     ]

#     layer_names = [
#         "nuclei",
#         "lyso",
#         "mitochondria",
#         "golgi",
#         "peroxisome",
#         "er",
#         "LD_body",
#         "cell",
#         "cytoplasm_mask",
#     ]
#     # TODO: pack outputs into something napari readable
#     img_out = np.stack(img_layers, axis=0)
#     return (img_out, layer_names)


# def batch_process_all_czi(data_root_path, source_dir: Union[Path, str] = "raw"):
#     """ """
#     # linearly unmixed ".czi" files are here
#     data_path = data_root_path / "raw"
#     im_type = ".czi"
#     # get the list of all files
#     img_file_list = list_image_files(data_path, im_type)
#     files_generated = []
#     for czi_file in img_file_list:
#         out_fn = process_czi_image(czi_file)
#         files_generated.append(out_fn)

#     print(f"generated {len(files_generated)} ")
#     return files_generated


# def process_czi_image(czi_file_name, data_root_path):
#     """wrapper for processing"""

#     img_data, meta_dict = read_czi_image(czi_file_name)
#     # # get some top-level info about the RAW data
#     # channel_names = meta_dict['name']
#     # img = meta_dict['metadata']['aicsimage']
#     # scale = meta_dict['scale']
#     # channel_axis = meta_dict['channel_axis']

#     inferred_organelles, layer_names, optimal_Z = fixed_infer_organelles(img_data)
#     out_file_n = export_inferred_organelle(inferred_organelles, layer_names, meta_dict, data_root_path)

#     ## TODO:  collect stats...

#     return out_file_n


# def stack_organelle_objects(
#     cellmask,
#     nuclei_object,
#     cytoplasm_mask,
#     lyso_object,
#     mito_object,
#     golgi_object,
#     peroxi_object,
#     er_object,
#     LD_object,
# ) -> np.ndarray:
#     """wrapper to stack the inferred objects into a single numpy.ndimage"""
#     img_layers = [
#         cellmask,
#         nuclei_object,
#         cytoplasm_mask,
#         lyso_object,
#         mito_object,
#         golgi_object,
#         peroxi_object,
#         er_object,
#         LD_object,
#     ]
#     return np.stack(img_layers, axis=0)


# def stack_organelle_layers(*layers) -> np.ndarray:
#     """wrapper to stack the inferred objects into a single numpy.ndimage"""

#     return np.stack(layers, axis=0)

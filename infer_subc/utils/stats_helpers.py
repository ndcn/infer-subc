import numpy as np
from typing import Any, List, Union
from pathlib import Path
import itertools 
import time

from infer_subc.core.img import apply_mask

import pandas as pd

from infer_subc.utils.stats import ( get_contact_metrics_3D, 
                    get_org_metrics_3D, 
                    get_simple_stats_3D, 
                    get_XY_dist_metrics, 
                    get_Z_dist_metrics, 
                    _assert_uint16_labels,
                    get_region_metrics_3D)

from infer_subc.utils.batch import list_image_files, find_segmentation_tiff_files
from infer_subc.core.file_io import read_czi_image, read_tiff_image

from infer_subc.constants import organelle_to_colname, NUC_CH, GOLGI_CH, PEROX_CH


organelle_to_colname = {"nuc":"NU", "lyso": "LY", "mito":"MT", "golgi":"GL", "perox":"PR", "ER":"ER", "LD":"LD", "cell":"CM", "cyto":"CY", "nucleus": "N1","nuclei":"NU",}

def make_all_metrics_tables(
    organelle_names: List[str],
    organelles: List[np.ndarray],
    intensities: List[np.ndarray],
    region_names: List[str],
    regions: List[np.ndarray],
    dist_centering_obj:np.ndarray, 
    mask:np.ndarray,
    source_file: str,
    scale: Union[tuple,None] = None,
    n_XY_bins: Union[int,None] = None,
    n_zernike: Union[int,None] = None,
    include_contact_dist: bool = True
) -> int:
    """
    get summary and all cross stats between organelles `a` and `b`
    calls `get_summary_stats_3D`
    order of intensity measurements in the regions dataframe is the same as the order of organelle names provided in the organelle_names list
    """
    start = time.time()
    count = 0
    ####################
    # collect individual organelle measurements, 
    # the organelle contacts associated the that organelle, 
    # and total organelle distribution measurements
    # for each organelle
    ####################
    org_tabs = []
    rps = []
    dist_tabs = []
    XY_bins = []

    # zern_bins = []
    for j, target in enumerate(organelle_names):
        org_img = intensities[j]
        if target == 'ER':
            org_obj = (organelles[j] > 0).astype(np.uint16)
        else:
            org_obj = organelles[j]

        ####################
        # measure individual organelle and cell region metrics
        ####################
        org_metrics, rp = get_org_metrics_3D(segmentation_img=org_obj, 
                                             intensity_img=org_img, 
                                             mask=mask,
                                             scale=scale)
        org_metrics.insert(loc=0,column='organelle',value=target)

        ####################
        # collect organelles in contact with this organelle
        ####################
        for i, nmi in enumerate(organelle_names):
            if i != j:
                if target == 'ER':
                    b = (organelles[i] > 0).astype(np.uint16)
                else:
                    b = organelles[i]
            
                ov = []
                b_labs = []
                labs = []
                for idx, lab in enumerate(org_metrics["label"]):
                    xyz = tuple(rp[idx].coords.T)
                    cmp_org = b[xyz]
                    
                    # total area (in voxels or real world units) where these two orgs overlap within the cell
                    if scale != None:
                        overlap = sum(cmp_org > 0)*scale[0]*scale[1]*scale[2]
                    else:
                        # total number of overlapping pixels
                        overlap = sum(cmp_org > 0)
                        # overlap?
                    
                    # which b organelles are involved in that overlap
                    labs_b = cmp_org[cmp_org > 0]
                    b_js = np.unique(labs_b).tolist()

                    # if overlap > 0:
                    labs.append(lab) # labs.append(lab)
                    ov.append(overlap)
                    b_labs.append(b_js)
                org_metrics[f"{nmi}_overlap"] = ov
                org_metrics[f"{nmi}_labels"] = b_labs 

        ####################
        # measure organelle distribution 
        ####################
        XY_org_distribution ,zernike_org_distribution, XY_bin_masks = get_XY_dist_metrics(cellmask_obj=mask, # TODO: add output for zernike masks
                                                                 nuclei_obj=dist_centering_obj,
                                                                 organelle_obj=org_obj,
                                                                 organelle_name=target,
                                                                 num_bins=n_XY_bins,
                                                                 zernike_degrees=n_zernike)

        Z_org_distribution = get_Z_dist_metrics(cellmask_obj=mask, 
                                            organelle_obj=org_obj,
                                            organelle_name=target,
                                            nuclei_obj=dist_centering_obj)
      
        org_distribution_metrics = pd.merge(XY_org_distribution, zernike_org_distribution,on=["organelle"])
        org_distribution_metrics = pd.merge(org_distribution_metrics, Z_org_distribution,on=["organelle"])

        org_tabs.append(org_metrics)
        rps.append(rp)
        dist_tabs.append(org_distribution_metrics)
        XY_bins.append(XY_bin_masks)
        # zern_bins.append(zernike_masks)
    
    ####################
    # measure individual organelle and cell region metrics
    ####################
    raw_image = np.stack(intensities)
    region_metrics = get_region_metrics_3D(region_segs=regions, 
                                        region_names=region_names,
                                        intensity_img=raw_image,
                                        mask=mask)

    ####################
    # collect non-redundant contact metrics
    #  TODO: figure out how to add the shell measurements as new columns, not rows    
    ####################
    contact_combos = list(itertools.combinations(organelle_names, 2))
    contact_tabs = []
    for pair in contact_combos:
        a_name = pair[0]
        b_name = pair[1]

        i = organelle_names.index(a_name)
        j = organelle_names.index(b_name)

        a = organelles[i] # org_obj
        b = organelles[j]

        if include_contact_dist == True:
            contact_tab, contact_XY_dist, contact_Zern_dist, contact_Z_dist = get_contact_metrics_3D(a, a_name, b, b_name, mask, 
                                                                                                     False, True, dist_centering_obj)
            
            contact_dist_metrics = pd.merge(contact_XY_dist, contact_Zern_dist,on=["organelle"])
            contact_dist_metrics = pd.merge(contact_dist_metrics, contact_Z_dist,on=["organelle"])
        else:
            contact_tab = get_contact_metrics_3D(a, a_name, b, b_name, mask, False)
            
        contact_tabs.append(contact_tab)
        dist_tabs.append(contact_dist_metrics)

    ####################
    # combine all tabs into one table per type:
    ####################
    final_org_tab = pd.concat(org_tabs, ignore_index=True)
    final_org_tab.insert(loc=0,column='image_name',value=source_file.stem )

    final_contact_tab = pd.concat(contact_tabs, ignore_index=True)
    final_contact_tab.insert(loc=0,column='image_name',value=source_file.stem )

    combined_dist_tab = pd.concat(dist_tabs, ignore_index=True)
    combined_dist_tab.insert(loc=0,column='image_name',value=source_file.stem )

    region_metrics.insert(loc=0,column='image_name',value=source_file.stem )

    end = time.time()
    print(f"It took {(end-start)/60} minutes to quantify one image.")
    return final_org_tab, final_contact_tab, combined_dist_tab, region_metrics


# def make_organelle_stat_tables(
#     organelle_names: List[str],
#     organelles: List[np.ndarray],
#     intensities: List[np.ndarray],
#     nuclei_obj:np.ndarray, 
#     cellmask_obj:np.ndarray,
#     organelle_mask: np.ndarray, 
#     out_data_path: Path, 
#     source_file: str,
#     n_rad_bins: Union[int,None] = None,
#     n_zernike: Union[int,None] = None,
# ) -> int:
#     """
#     get summary and all cross stats between organelles `a` and `b`
#     calls `get_org_metrics_3D`
#     """
#     count = 0
#     org_stats_tabs = []
#     for j, target in enumerate(organelle_names):
#         org_img = intensities[j]        
#         org_obj = _assert_uint16_labels(organelles[j])

#         # A_stats_tab, rp = get_simple_stats_3D(A,mask)
#         a_stats_tab, rp = get_org_metrics_3D(org_obj, org_img, organelle_mask)
#         a_stats_tab.insert(loc=0,column='organelle',value=target )
#         a_stats_tab.insert(loc=0,column='ID',value=source_file.stem )

#         # add the touches for all other organelles
#         # loop over Bs
#         merged_tabs = []
#         for i, nmi in enumerate(organelle_names):
#             if i != j:
#                 # get overall stats of intersection
#                 # print(f"  b = {nmi}")
#                 count += 1
#                 # add the list of touches
#                 b = _assert_uint16_labels(organelles[i])

#                 ov = []
#                 b_labs = []
#                 labs = []
#                 for idx, lab in enumerate(a_stats_tab["label"]):  # loop over A_objects
#                     xyz = tuple(rp[idx].coords.T)
#                     cmp_org = b[xyz]
                    
#                     # total number of overlapping pixels
#                     overlap = sum(cmp_org > 0)
#                     # overlap?
#                     labs_b = cmp_org[cmp_org > 0]
#                     b_js = np.unique(labs_b).tolist()

#                     # if overlap > 0:
#                     labs.append(lab)
#                     ov.append(overlap)
#                     b_labs.append(b_js)

#                 cname = organelle_to_colname[nmi]
#                 # add organelle B columns to A_stats_tab
#                 a_stats_tab[f"{cname}_overlap"] = ov
#                 a_stats_tab[f"{cname}_labels"] = b_labs  # might want to make this easier for parsing later

#                 #####  2  ###########
#                 # get cross_stats

#                 cross_tab = get_contact_metrics_3D(org_obj, b, organelle_mask) 
#                 shell_cross_tab = get_contact_metrics_3D(org_obj, b, organelle_mask, use_shell_a=True)
                            
#                 # cross_tab["organelle_b"]=nmi
#                 # shell_cross_tab["organelle_b"]=nmi
#                 #  Merge cross_tabs and shell_cross_tabs 
#                 # merged_tab = pd.merge(cross_tab,shell_cross_tab, on="label_")
#                 merged_tab = pd.concat([cross_tab,shell_cross_tab])
#                 merged_tab.insert(loc=0,column='organelle_b',value=nmi )

#                 merged_tabs.append( merged_tab )


#         #  Now append the 
#         # csv_path = out_data_path / f"{source_file.stem}-{target}_shellX{nmi}-stats.csv"
#         # e_stats_tab.to_csv(csv_path)
#         # stack these tables for each organelle
#         crossed_tab = pd.concat(merged_tabs)
#         # csv_path = out_data_path / f"{source_file.stem}-{target}X{nmi}-stats.csv"
#         # stats_tab.to_csv(csv_path)
#         crossed_tab.insert(loc=0,column='organelle',value=target )
#         crossed_tab.insert(loc=0,column='ID',value=source_file.stem )

#         # now get radial stats
#         rad_stats,z_stats, _ = get_radial_stats(        
#                 cellmask_obj,
#                 organelle_mask,
#                 org_obj,
#                 org_img,
#                 target,
#                 nuclei_obj,
#                 n_rad_bins,
#                 n_zernike
#                 )

#         d_stats = get_depth_stats(        
#                 cellmask_obj,
#                 organelle_mask,
#                 org_obj,
#                 org_img,
#                 target,
#                 nuclei_obj
#                 )
      
#         proj_stats = pd.merge(rad_stats, z_stats,on=["organelle","mask"])
#         proj_stats = pd.merge(proj_stats, d_stats,on=["organelle","mask"])
#         proj_stats.insert(loc=0,column='ID',value=source_file.stem )


#         # write out files... 
#         # org_stats_tabs.append(A_stats_tab)
#         csv_path = out_data_path / f"{source_file.stem}-{target}-stats.csv"
#         a_stats_tab.to_csv(csv_path)

#         csv_path = out_data_path / f"{source_file.stem}-{target}-cross-stats.csv"
#         crossed_tab.to_csv(csv_path)

#         csv_path = out_data_path / f"{source_file.stem}-{target}-proj-stats.csv"
#         proj_stats.to_csv(csv_path)

#         count += 1

#     print(f"dumped {count}x3 organelle stats ({organelle_names}) csvs")
#     return count


def _batch_process_quantification(out_file_prefix: str,
                                  seg_path: Union[Path,str],
                                  out_path: Union[Path, str], 
                                  raw_path: Union[Path,str], 
                                  raw_file_type: str,
                                  organelle_names: List[str], 
                                  organelle_chs: List[int], 
                                  region_names: List[str],
                                  seg_suffix: Union[str, None] = None,
                                  scale: bool = True,
                                  n_XY_bins: Union[int,None] = 5,
                                  n_zernike: Union[int,None] = 9,
                                  include_contact_dist: bool = True
                                    ) -> int :
    """  
    TODO: add loggin instead of printing
        append tiffcomments with provenance

    organelle_names - should match the file suffix; I think the order will specify the order in the csv files
        
    """
    start = time.time()
    if isinstance(raw_path, str): raw_path = Path(raw_path)
    if isinstance(seg_path, str): seg_path = Path(seg_path)
    if isinstance(out_path, str): out_path = Path(out_path)
    
    img_file_list = list_image_files(raw_path,raw_file_type)

    if not Path.exists(out_path):
        Path.mkdir(out_path)
        print(f"making {out_path}")
    
    count = 0

    org_tabs = []
    contact_tabs = []
    dist_tabs = []
    region_tabs = []
    
    segs_to_collect = organelle_names + region_names

    if scale == True:
        for img_f in img_file_list:
            count = count + 1
            filez = find_segmentation_tiff_files(img_f, segs_to_collect, seg_path, seg_suffix)
            img_data, meta_dict = read_czi_image(filez["raw"])
            scale = meta_dict['scale'] # this will only work if the scale can be pulled from the metadata this way

            # create intensities from raw as list
            intensities = [img_data[ch] for ch in organelle_chs]

            # load regions
            regions = []
            for name in region_names:
                mask = read_tiff_image(filez[name])
                regions.append(mask)

            nuc_mask = regions[region_names.index('nuc')]
            cell_mask = regions[region_names.index('cell')]

            # load organelles as list
            organelles = [read_tiff_image(filez[org]) for org in organelle_names]

            if scale == True:
                org_metrics, contact_metrics, dist_metrics, region_metrics = make_all_metrics_tables(organelle_names=organelle_names,
                                                                                    organelles=organelles,
                                                                                    intensities=intensities,
                                                                                    regions=regions,
                                                                                    region_names=region_names,
                                                                                    dist_centering_obj=nuc_mask, 
                                                                                    mask=cell_mask,
                                                                                    source_file=img_f,
                                                                                    scale=scale,
                                                                                    n_XY_bins=n_XY_bins,
                                                                                    n_zernike=n_zernike,
                                                                                    include_contact_dist=include_contact_dist)
            else:
                org_metrics, contact_metrics, dist_metrics, region_metrics = make_all_metrics_tables(organelle_names=organelle_names,
                                                                                    organelles=organelles,
                                                                                    intensities=intensities,
                                                                                    regions=regions,
                                                                                    region_names=region_names,
                                                                                    dist_centering_obj=nuc_mask, 
                                                                                    mask=cell_mask,
                                                                                    source_file=img_f,
                                                                                    n_XY_bins=n_XY_bins,
                                                                                    n_zernike=n_zernike,
                                                                                    include_contact_dist=include_contact_dist)
            org_tabs.append(org_metrics)
            contact_tabs.append(contact_metrics)
            dist_tabs.append(dist_metrics)
            region_tabs.append(region_metrics)
            end2 = time.time()
            print(f"Completed processing for {count} images in {(end2-start)/60} mins.")

    final_org = pd.concat(org_tabs, ignore_index=True)
    final_contact = pd.concat(contact_tabs, ignore_index=True)
    final_dist = pd.concat(dist_tabs, ignore_index=True)
    final_region = pd.concat(region_tabs, ignore_index=True)

    org_csv_path = out_path / f"{out_file_prefix}organelles.csv"
    final_org.to_csv(org_csv_path)

    contact_csv_path = out_path / f"{out_file_prefix}contacts.csv"
    final_contact.to_csv(contact_csv_path)

    dist_csv_path = out_path / f"{out_file_prefix}distributions.csv"
    final_dist.to_csv(dist_csv_path)

    region_csv_path = out_path / f"{out_file_prefix}regions.csv"
    final_region.to_csv(region_csv_path)

    end = time.time()
    print(f"Quantification for {count} files is COMPLETE! Files saved to '{out_data_path}'.")
    print(f"It took {(end - start)/60} minutes to quantify these files.")
    return count


# def dump_all_stats_tables(int_path: Union[Path,str], 
#                    out_path: Union[Path, str], 
#                    raw_path: Union[Path,str], 
#                    organelle_names: List[str]= ["nuclei","golgi","peroxi"], 
#                    organelle_chs: List[int]= [NUC_CH,GOLGI_CH, PEROX_CH], 
#                     ) -> int :
#     """  
#     TODO: add loggin instead of printing
#         append tiffcomments with provenance
#     """

    
#     if isinstance(raw_path, str): raw_path = Path(raw_path)
#     if isinstance(int_path, str): int_path = Path(int_path)
#     if isinstance(out_path, str): out_path = Path(out_path)
    
#     img_file_list = list_image_files(raw_path,".czi")

#     if not Path.exists(out_path):
#         Path.mkdir(out_path)
#         print(f"making {out_path}")
        
#     for img_f in img_file_list:
#         filez = find_segmentation_tiff_files(img_f, organelle_names, int_path)
#         img_data,meta_dict = read_czi_image(filez["raw"])

#         # load organelles and masks
#         cyto_mask = read_tiff_image(filez["cyto"])
#         cellmask_obj = read_tiff_image(filez["cell"])



#         # create intensities from raw as list
#         intensities = [img_data[ch] for ch in organelle_chs]

#         # load organelles as list
#         organelles = [read_tiff_image(filez[org]) for org in organelle_names]
        
#         #get mask (cyto_mask)
#         nuclei_obj = organelles[ organelle_names.index("nuc") ]

#         n_files = make_organelle_stat_tables(organelle_names, 
#                                       organelles,
#                                       intensities, 
#                                       nuclei_obj,
#                                       cellmask_obj,
#                                       cyto_mask, 
#                                       out_path, 
#                                       img_f,
#                                       n_rad_bins=5,
#                                       n_zernike=9)

#     return n_files





# These are probably not to be used
def dump_shell_cross_stats(
    organelle_names: List[str], organelles: List[np.ndarray], mask: np.ndarray, out_data_path: Path, source_file: str
) -> int:
    """
    get all cross stats between organelles `a` and `b`, and "shell of `a`" and `b`.   "shell" is the boundary of `a`
    calls `get_contact_metrics_3D`
    """
    count = 0
    for j, target in enumerate(organelle_names):
        # print(f"getting stats for org_obj = {target}")
        org_obj = organelles[j]
        # loop over Bs
        for i, nmi in enumerate(organelle_names):
            if i != j:
                # get overall stats of intersection
                # print(f"  X {nmi}")
                b = organelles[i]
                stats_tab = get_contact_metrics_3D(org_obj, b, mask)
                csv_path = out_data_path / f"{source_file.stem}-{target}X{nmi}-stats.csv"
                stats_tab.to_csv(csv_path)

                e_stats_tab = get_contact_metrics_3D(org_obj, b, mask, use_shell_a=True)
                csv_path = out_data_path / f"{source_file.stem}-{target}_shellX{nmi}-stats.csv"
                e_stats_tab.to_csv(csv_path)

                count += 1
    print(f"dumped {count} x2 organelle cross-stats csvs")
    return count





def load_summary_stats_csv(in_path: Path) -> pd.DataFrame:
    """ helper to load the summary stats csv: summary-stats.csv
    returns pandas DataFrame """
    csv_path = in_path / f"summary-stats.csv"
    summary_df = pd.read_csv(csv_path, index_col=0)
    # need to convert columns *_labels
    list_cols = [col for col in summary_df.columns if "labels" in col] #if col.contains("label")
    summary_df = fix_int_list_cols(summary_df,list_cols)
    return summary_df


def load_summary_proj_stats_csv(in_path: Path) -> pd.DataFrame:
    """ helper to load summary projection stats csv: summary-proj-stats.csv
    returns pandas DataFrame """
    obj_cols =  ['ID', 'organelle','mask','radial_n_bins','n_z']  # leave alone
    str_cols = [ 'radial_bins']
    int_cols = ['radial_cm_vox_cnt', 'radial_org_vox_cnt', 'radial_org_intensity', 'radial_n_pix','zernike_n', 'zernike_m', 'z','z_cm_vox_cnt','z_org_vox_cnt', 'z_org_intensity', 'z_nuc_vox_cnt']
    float_cols = ['radial_cm_cv', 'radial_org_cv', 'radial_img_cv','zernike_cm_mag', 'zernike_cm_phs','zernike_obj_mag', 'zernike_obj_phs', 'zernike_nuc_mag','zernike_nuc_phs', 'zernike_img_mag']

    csv_path = in_path / f"summary-proj-stats.csv"
    proj = pd.read_csv(csv_path, index_col=0)
    proj = fix_str_list_cols(proj, str_cols)
    proj = fix_int_list_cols(proj, int_cols)
    proj = fix_float_list_cols(proj, float_cols)
    return proj
        

def load_summary_cross_stats_csv(in_path: Path) -> pd.DataFrame:
    """ helper to load summary cross- stats csv: summary-cross-stats.csv
    returns pandas DataFrame """

    csv_path = in_path / f"summary-cross-stats.csv"
    summary_df = pd.read_csv(csv_path, index_col=0)

    list_cols = [col for col in summary_df.columns if "label" in col] #if col.contains("label")
    str_list_cols = [col for col in list_cols if "__" in col]
    int_list_cols = [col for col in list_cols if "__" not in col]

    summary_df = fix_str_list_cols(summary_df,str_list_cols)
    summary_df = fix_int_list_cols(summary_df,int_list_cols)

    return summary_df
    


# for a list of "prefixes"  collect stats + cross stats masked by cytosol (including nuclei masked by cellmask)

def summarize_by_id(stats_in:pd.DataFrame,agg_fn: List) -> pd.DataFrame:
    """ 
    """
    summary = stats_in.groupby(['ID']).agg(agg_fn)
    summary.columns = ["_".join(col_name).rstrip('_') for col_name in summary.columns.to_flat_index()]
    return summary



def create_stats_summary(summary_df:pd.DataFrame) -> pd.DataFrame:
    """
    """
    column_names = summary_df.columns

    def frac(x):
        return (x>0).sum()/x.count() 

    math_cols = ['ID', 'mean_intensity',
        'standard_deviation_intensity',
        'min_intensity','max_intensity', 'equivalent_diameter',
        'euler_number', 'extent']
    vol_cols = ['ID','volume']
    overlap_cols = ['ID'] + [col for col in column_names if col.endswith('_overlap')]
    labels_cols = ['ID'] + [col for col in column_names if col.endswith('_labels')]
   
    agg_func_math = ['sum', 'mean', 'median', 'min', 'max', 'std','count']
    agg_func_overlap = ['sum', 'mean', 'median','count',frac]
    agg_func_labels = ['sum']
    agg_func_vol = ['sum', 'mean', 'median', 'min', 'max', 'std', 'var']

    math_summary = summarize_by_id( summary_df[math_cols] , agg_func_math)
    
    # label_stats = fix_list_col(summary_df[labels_cols])
    label_summary = summarize_by_id( summary_df[labels_cols] , agg_func_labels)
    overlap_summary = summarize_by_id( summary_df[overlap_cols] ,agg_func_overlap)
    vol_summary = summarize_by_id( summary_df[vol_cols] , agg_func_vol)
    result = pd.concat([math_summary, vol_summary, overlap_summary, label_summary], axis=1)

    result.insert(loc=0,column="ID",value=result.index)

    return result


def summarize_by_group(stats_in:pd.DataFrame, grp_col:list, agg_fn:list) -> pd.DataFrame:
    """ 
    """
    summary = stats_in.reset_index(drop=True).groupby(grp_col).agg(agg_fn)
    summary.columns = ["_".join(col_name).rstrip('_') for col_name in summary.columns.to_flat_index()]
    return summary


def create_cross_stats_summary(summary_df:pd.DataFrame) -> pd.DataFrame:
    """
    """
    # dropped_cols = ['centroid-0', 'centroid-1', 'centroid-2', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5']
    # cross_cols = ['ID', 'organelle', 'organelle_b', 'shell', 'label_', 'label', 'volume',
    #    'equivalent_diameter','surface_area', 'label_a', 'label_b']

    group_cols = ['ID','organelle_b', 'shell']
    id_cols = ['label_','label_a', 'label_b'] 
    math_cols = ['volume','equivalent_diameter','surface_area']

    def lst(x):
        return x.to_list()
       
    agg_func_math = ['sum', 'mean', 'median', 'min', 'max', 'std','count']
    agg_func_id = [lst]

    math_summary = summarize_by_group( summary_df[group_cols + math_cols],group_cols, agg_func_math)

    id_summary = summarize_by_group( summary_df[group_cols + id_cols],group_cols, agg_func_id)

    result = pd.concat([math_summary, id_summary], axis=1)
    return result

    # now 

def summarize_cross_stats(summary_df:pd.DataFrame) -> pd.DataFrame:
    """
    """
    # get shell
    shell_summary_i = create_cross_stats_summary(summary_df.loc[summary_df["shell"] == True]).reset_index().drop("shell", axis = 1).add_prefix("shell_")
    # rename shell_ID to ID
    shell_summary_i = shell_summary_i.rename(columns={"shell_ID":"ID","shell_organelle_b":"organelle_b"})

    # get non-shell
    summary_i = create_cross_stats_summary(summary_df.loc[summary_df["shell"] == False]).reset_index().drop("shell", axis = 1)

    summary_i = summary_i.merge(shell_summary_i,on=["ID","organelle_b"])

    return summary_i


def pivot_cross_stats(summary_df:pd.DataFrame) -> pd.DataFrame:
    """
    """
    xstat_df = pd.DataFrame()
    org_bs = summary_df["organelle_b"].unique()
    for i,org_b in enumerate(org_bs):
        org_i = summary_df.loc[summary_df["organelle_b"] == org_b]

        # get shell
        shell_summary_i = create_cross_stats_summary(org_i.loc[org_i["shell"] == False]).reset_index().drop("shell", axis = 1).add_prefix("shell_")
        # rename shell_ID to ID
        shell_summary_i = shell_summary_i.rename(columns={"shell_ID":"ID","shell_organelle_b":"organelle_b"})
        # get non-shell
        summary_i = create_cross_stats_summary(org_i.loc[org_i["shell"] == False]).reset_index().drop("shell", axis = 1)
        col_orgb = organelle_to_colname[org_b]

        summary_i = summary_i.merge(shell_summary_i,on=["ID","organelle_b"]).drop("organelle_b", axis=1).add_suffix(f"_{col_orgb}")
        if i>0:
            xstat_df = pd.concat([xstat_df,summary_i], axis=1)
        else:
            xstat_df = summary_i
            
    id_cols = [col for col in xstat_df.columns if "ID" in col]
    IDcol = xstat_df[id_cols[0]]
    xstat_df = xstat_df.drop(id_cols, axis=1)
    xstat_df.insert(loc=0,column="ID",value=IDcol)

    return xstat_df


def fix_int_list_cols(in_df:pd.DataFrame, list_cols) -> pd.DataFrame:
    """ 
    """

    def _str_to_list(x):
        if x == '[]':
            return list()
        elif isinstance(x,float): # catch nans
            return x
        else:
            xstr = x.strip("[]").replace("'", "").split(", ")
        return [int(float(x)) for x in xstr]
        
    out_df = pd.DataFrame() 
    for col in in_df.columns:    
        out_df[col] = in_df[col].apply(_str_to_list) if col in list_cols else in_df[col]
    return out_df

def fix_float_list_cols(in_df:pd.DataFrame, list_cols) -> pd.DataFrame:
    """ 
    """
    def _str_to_list(x):
        if x == '[]':
            return list()
        elif isinstance(x,float): # catch nans
            return x
        else:
            xstr = x.strip("[]").replace("'", "").split(", ")
        return [float(x) for x in xstr]
        
    out_df = pd.DataFrame() 
    for col in in_df.columns:    
        out_df[col] = in_df[col].apply(_str_to_list) if col in list_cols else in_df[col]
    return out_df

def fix_str_list_cols(in_df:pd.DataFrame, list_cols) -> pd.DataFrame:
    """ 
    """
    def _str_to_list(x):
        if x == '[]':
            return list()
        elif isinstance(x,float): # catch nans
            return x
        else:
            xstr = x.strip("[]").replace("'", "").split(", ")
        return [x for x in xstr]
        
    out_df = pd.DataFrame() 
    for col in in_df.columns:    
        out_df[col] = in_df[col].apply(_str_to_list) if col in list_cols else in_df[col]
    return out_df


def load_stats_csv(in_path: Path, img_id: str, target_org: str) -> pd.DataFrame:
    """ helper to load the basic stats csv: `img_id`-`target_organelle` -stats.csv
    returns pandas DataFrame """
    csv_path = in_path / f"{img_id}-{target_org}-stats.csv"
    stats = pd.read_csv(csv_path, index_col=0,dtype={"ID":str,"organelle":str})
    # need to convert columns *_labels
    list_cols = [col for col in stats.columns if col.endswith('_labels')]
    stats = fix_int_list_cols(stats,list_cols)
    return stats
        

def load_proj_stats_csv(in_path: Path, img_id: str, target_org: str) -> pd.DataFrame:
    """ helper to load  the projection stats csv: `img_id`-`target_organelle` -proj-stats.csv
    returns pandas DataFrame """
    # obj_cols =  ['ID', 'organelle','radial_n_bins','n_z']  # leave alone
    # str_cols = [ 'radial_bins']
    int_cols = ['radial_cm_vox_cnt', 'radial_org_vox_cnt', 'radial_org_intensity', 'radial_n_pix','zernike_n', 'zernike_m', 'z','z_cm_vox_cnt','z_org_vox_cnt', 'z_org_intensity', 'z_nuc_vox_cnt']
    float_cols = ['radial_cm_cv', 'radial_org_cv', 'radial_img_cv','zernike_cm_mag', 'zernike_cm_phs','zernike_obj_mag', 'zernike_obj_phs', 'zernike_nuc_mag','zernike_nuc_phs', 'zernike_img_mag']

    csv_path = in_path / f"{img_id}-{target_org}-proj-stats.csv"
    proj = pd.read_csv(csv_path, index_col=0)
    proj['radial_bins'] = proj['radial_bins'].values.squeeze().tolist()
    # proj = fix_str_list_cols(proj, str_cols)
    proj = fix_int_list_cols(proj, int_cols)
    proj = fix_float_list_cols(proj, float_cols)
    return proj
        

def load_cross_stats_csv(in_path: Path, img_id: str, target_org: str) -> pd.DataFrame:
    """ helper to load  the cross- stats csv: `img_id`-`target_organelle` -cross-stats.csv
    returns pandas DataFrame """
    csv_path = in_path / f"{img_id}-{target_org}-cross-stats.csv"
    cross = pd.read_csv(csv_path, index_col=0)
    return cross


def summarize_organelle_stats(int_path: Union[Path,str], 
                              organelle_names: List[str]= ["nuclei","golgi","peroxi"]):
    """  
    """
    # write out files... 

    if isinstance(int_path, str): int_path = Path(int_path)


    all_stats_df = pd.DataFrame()
    all_cross_stats_df = pd.DataFrame()
    all_proj_stats_df = pd.DataFrame()
    
    for target in organelle_names:
        stat_file_list = sorted( int_path.glob(f"*{target}-stats.csv") )

        stats_df = pd.DataFrame()
        cross_stats_df = pd.DataFrame()
        proj_stats_df = pd.DataFrame()

        for stats_f in stat_file_list:
            stem = stats_f.stem.split("-")[0]
            # stats load the csv
            stats = load_stats_csv(int_path,stem, target)
            # projection stats
            proj = load_proj_stats_csv(int_path,stem, target)
            # cross stats
            cross = load_cross_stats_csv(int_path,stem, target)

            stats_df = pd.concat([stats_df,stats],axis=0, join='outer')
            proj_stats_df = pd.concat([proj_stats_df,proj],axis=0, join='outer')
            cross_stats_df = pd.concat([cross_stats_df,cross],axis=0, join='outer')
        

        ## maybe merge into all the possible files?
        # summary_df = pd.DataFrame(index=[f.stem.split("-")[0] for f in stat_file_list])
        # cross_stats_df = pd.DataFrame(index=[f.stem.split("-")[0] for f in stat_file_list])
        # proj_stats_df = pd.DataFrame(index=[f.stem.split("-")[0] for f in stat_file_list])

        summary_df = create_stats_summary(stats_df)
        summary_df.insert(loc=1,column="organelle",value=target)
        cross_summary_df = summarize_cross_stats(cross_stats_df)
        ## cross_summary_df = pivot_cross_stats(cross_stats_df)  #makes a wide version... but has a bug
        cross_summary_df.insert(loc=1,column="organelle",value=target)

        all_stats_df = pd.concat([all_stats_df,summary_df],axis=0)
        all_proj_stats_df = pd.concat([all_proj_stats_df,proj_stats_df],axis=0)
        all_cross_stats_df = pd.concat([all_cross_stats_df,cross_summary_df],axis=0)
    

    return all_stats_df, all_proj_stats_df, all_cross_stats_df
        



def dump_organelle_summary_tables(
                    int_path: Union[Path,str], 
                    out_path: Union[Path, str], 
                    organelle_names: List[str]= ["nuclei","golgi","peroxi"] ) -> int:
    """
    get summary and all cross stats between organelles `a` and `b`
    calls `get_org_metrics_3D`
    """

    if not Path.exists(out_path):
        Path.mkdir(out_path)
        print(f"making {out_path}")


    all_stats_df, all_proj_stats_df, all_cross_stats_df = summarize_organelle_stats( int_path, organelle_names )

    csv_path = out_path / f"summary-stats.csv"
    all_stats_df.to_csv(csv_path)

    csv_path = out_path / f"summary-proj-stats.csv"
    all_proj_stats_df.to_csv(csv_path)

    csv_path = out_path / f"summary-cross-stats.csv"
    all_cross_stats_df.to_csv(csv_path)

    return 1


def dump_stats(
    name: str,
    segmentation: np.ndarray,
    intensity_img: np.ndarray,
    mask: np.ndarray,
    out_data_path: Path,
    source_file: str,
) -> pd.DataFrame:
    """
    get summary stats of organelle only
    calls `get_org_metrics_3D`
    """
    stats_table, _ = get_org_metrics_3D(segmentation, intensity_img, mask)
    csv_path = out_data_path / f"{source_file.stem}-{name}-basic-stats.csv"
    stats_table.to_csv(csv_path)
    print(f"dumped {name} table to {csv_path}")

    return stats_table


# refactor to just to a target vs. list of probes
# for nuclei mask == cellmask
# for all oother mask == cytoplasm


def dump_projection_stats(
    organelle_names: List[str], 
    organelles: List[np.ndarray], 
    intensities: List[np.ndarray], 
    cellmask_obj:np.ndarray, 
    nuclei_obj:np.ndarray, 
    organelle_mask: np.ndarray, 
    out_data_path: Path, 
    source_file: str,
    n_rad_bins: Union[int,None] = None,
    n_zernike: Union[int,None] = None,
) -> int:
    """
    get all cross stats between organelles `a` and `b`, and "shell of `a`" and `b`.   "shell" is the boundary of `a`
    calls `get_proj_XYstats`  `get_proj_Zstats`
    """
    count = 0
    for j, organelle_name in enumerate(organelle_names):
       
        organelle_obj = organelles[j]
        organelle_img = intensities[j]

        rad_stats,z_stats, _ = get_radial_stats(        
                cellmask_obj,
                organelle_mask,
                organelle_obj,
                organelle_img,
                organelle_name,
                nuclei_obj,
                n_rad_bins,
                n_zernike
                )

        csv_path = out_data_path / f"{source_file.stem}-{organelle_name}-radial-stats.csv"
        rad_stats.to_csv(csv_path)

        csv_path = out_data_path / f"{source_file.stem}-{organelle_name}-zernike-stats.csv"
        z_stats.to_csv(csv_path)

        d_stats = get_depth_stats(        
                cellmask_obj,
                organelle_mask,
                organelle_obj,
                organelle_img,
                organelle_name,
                nuclei_obj
                )
        csv_path = out_data_path / f"{source_file.stem}-{organelle_name}-depth-stats.csv"
        d_stats.to_csv(csv_path)
        count += 1

    print(f"dumped {count}x3 projection stats csvs")
    return count
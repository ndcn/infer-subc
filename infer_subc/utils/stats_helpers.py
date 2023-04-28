import numpy as np
from typing import Any, List, Union
from pathlib import Path

from infer_subc.core.img import apply_mask

import pandas as pd

from .stats import ( get_aXb_stats_3D, 
                    get_summary_stats_3D, 
                    get_simple_stats_3D, 
                    get_radial_stats, 
                    get_depth_stats, 
                    _assert_uint16_labels )

from ..constants import organelle_to_colname

def make_organelle_stat_tables(
    organelle_names: List[str],
    organelles: List[np.ndarray],
    intensities: List[np.ndarray],
    nuclei_obj:np.ndarray, 
    cellmask_obj:np.ndarray,
    organelle_mask: np.ndarray, 
    out_data_path: Path, 
    source_file: str,
    n_rad_bins: Union[int,None] = None,
    n_zernike: Union[int,None] = None,
) -> int:
    """
    get summary and all cross stats between organelles `a` and `b`
    calls `get_summary_stats_3D`
    """
    count = 0
    org_stats_tabs = []
    for j, target in enumerate(organelle_names):
        org_img = intensities[j]        
        org_obj = _assert_uint16_labels(organelles[j])

        # A_stats_tab, rp = get_simple_stats_3D(A,mask)
        a_stats_tab, rp = get_summary_stats_3D(org_obj, org_img, organelle_mask)
        a_stats_tab.insert(loc=0,column='organelle',value=target )
        a_stats_tab.insert(loc=0,column='ID',value=source_file.stem )

        # add the touches for all other organelles
        # loop over Bs
        merged_tabs = []
        for i, nmi in enumerate(organelle_names):
            if i != j:
                # get overall stats of intersection
                # print(f"  b = {nmi}")
                count += 1
                # add the list of touches
                b = _assert_uint16_labels(organelles[i])

                ov = []
                b_labs = []
                labs = []
                for idx, lab in enumerate(a_stats_tab["label"]):  # loop over A_objects
                    xyz = tuple(rp[idx].coords.T)
                    cmp_org = b[xyz]
                    
                    # total number of overlapping pixels
                    overlap = sum(cmp_org > 0)
                    # overlap?
                    labs_b = cmp_org[cmp_org > 0]
                    b_js = np.unique(labs_b).tolist()

                    # if overlap > 0:
                    labs.append(lab)
                    ov.append(overlap)
                    b_labs.append(b_js)

                cname = organelle_to_colname[nmi]
                # add organelle B columns to A_stats_tab
                a_stats_tab[f"{cname}_overlap"] = ov
                a_stats_tab[f"{cname}_labels"] = b_labs  # might want to make this easier for parsing later

                #####  2  ###########
                # get cross_stats

                cross_tab = get_aXb_stats_3D(org_obj, b, organelle_mask) 
                shell_cross_tab = get_aXb_stats_3D(org_obj, b, organelle_mask, use_shell_a=True)
                            
                # cross_tab["organelle_b"]=nmi
                # shell_cross_tab["organelle_b"]=nmi
                #  Merge cross_tabs and shell_cross_tabs 
                # merged_tab = pd.merge(cross_tab,shell_cross_tab, on="label_")
                merged_tab = pd.concat([cross_tab,shell_cross_tab])
                merged_tab.insert(loc=0,column='organelle_b',value=nmi )

                merged_tabs.append( merged_tab )


        #  Now append the 
        # csv_path = out_data_path / f"{source_file.stem}-{target}_shellX{nmi}-stats.csv"
        # e_stats_tab.to_csv(csv_path)
        # stack these tables for each organelle
        crossed_tab = pd.concat(merged_tabs)
        # csv_path = out_data_path / f"{source_file.stem}-{target}X{nmi}-stats.csv"
        # stats_tab.to_csv(csv_path)
        crossed_tab.insert(loc=0,column='organelle',value=target )
        crossed_tab.insert(loc=0,column='ID',value=source_file.stem )

        # now get radial stats
        rad_stats,z_stats, _ = get_radial_stats(        
                cellmask_obj,
                organelle_mask,
                org_obj,
                org_img,
                target,
                nuclei_obj,
                n_rad_bins,
                n_zernike
                )

        d_stats = get_depth_stats(        
                cellmask_obj,
                organelle_mask,
                org_obj,
                org_img,
                target,
                nuclei_obj
                )
      
        proj_stats = pd.merge(rad_stats, z_stats,on=["organelle","mask"])
        proj_stats = pd.merge(proj_stats, d_stats,on=["organelle","mask"])
        proj_stats.insert(loc=0,column='ID',value=source_file.stem )

        # write out files... 
        # org_stats_tabs.append(A_stats_tab)
        csv_path = out_data_path / f"{source_file.stem}-{target}-stats.csv"
        a_stats_tab.to_csv(csv_path)

        csv_path = out_data_path / f"{source_file.stem}-{target}-cross-stats.csv"
        crossed_tab.to_csv(csv_path)

        csv_path = out_data_path / f"{source_file.stem}-{target}-proj-stats.csv"
        proj_stats.to_csv(csv_path)

        count += 1

    print(f"dumped {count}x3 organelle stats ({organelle_names}) csvs")
    return count



# These are probably not to be used
def dump_shell_cross_stats(
    organelle_names: List[str], organelles: List[np.ndarray], mask: np.ndarray, out_data_path: Path, source_file: str
) -> int:
    """
    get all cross stats between organelles `a` and `b`, and "shell of `a`" and `b`.   "shell" is the boundary of `a`
    calls `get_aXb_stats_3D`
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
                stats_tab = get_aXb_stats_3D(org_obj, b, mask)
                csv_path = out_data_path / f"{source_file.stem}-{target}X{nmi}-stats.csv"
                stats_tab.to_csv(csv_path)

                e_stats_tab = get_aXb_stats_3D(org_obj, b, mask, use_shell_a=True)
                csv_path = out_data_path / f"{source_file.stem}-{target}_shellX{nmi}-stats.csv"
                e_stats_tab.to_csv(csv_path)

                count += 1
    print(f"dumped {count} x2 organelle cross-stats csvs")
    return count




# def dump_organelle_stats(
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
#     calls `get_summary_stats_3D`
#     """
#     count = 0
#     org_stats_tabs = []
#     for j, target in enumerate(organelle_names):
#         # print(f"getting stats for A = {target}")
#         org_img = intensities[j]        
#         org_obj = _assert_uint16_labels(organelles[j])

#         # A_stats_tab, rp = get_simple_stats_3D(A,mask)
#         a_stats_tab, rp = get_summary_stats_3D(org_obj, org_img, organelle_mask)
#         a_stats_tab.insert(loc=0,column='organelle',value=target )
#         a_stats_tab.insert(loc=0,column='ID',value=source_file.stem )

#         # add the touches for all other organelles
#         # loop over Bs
#         merged_tabs = []
#         shell_cross_tabs = []
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

#                 cross_tab = get_aXb_stats_3D(org_obj, b, organelle_mask) 
#                 shell_cross_tab = get_aXb_stats_3D(org_obj, b, organelle_mask, use_shell_a=True)
                            
#                 cross_tab["organelle_b"]=nmi
#                 #  Merge cross_tabs and shell_cross_tabs 

#                 merged_tab = pd.merge(cross_tab,shell_cross_tab, on="label_")
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
   
        

#         crossed_tab.insert(loc=0,column='organelle',value=target )
#         crossed_tab.insert(loc=0,column='ID',value=source_file.stem )


#         # write out files... 
#         # org_stats_tabs.append(A_stats_tab)
#         csv_path = out_data_path / f"{source_file.stem}-{target}-stats.csv"
#         a_stats_tab.to_csv(csv_path)


#         csv_path = out_data_path / f"{source_file.stem}-{target}-cross-stats.csv"
#         crossed_tab.to_csv(csv_path)



#         csv_path = out_data_path / f"{source_file.stem}-{target}-cross-stats.csv"
#         crossed_tab.to_csv(csv_path)


#     print(f"dumped {count} organelle stats csvs")
#     return count


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
    calls `get_summary_stats_3D`
    """
    stats_table, _ = get_summary_stats_3D(segmentation, intensity_img, mask)
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
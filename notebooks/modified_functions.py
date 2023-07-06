from infer_subc.utils.stats import (get_normalized_distance_and_mask, 
                                    create_masked_Z_projection,
                                    zernike_polynomial,
                                    zernicke_stat)
import numpy as np
from skimage.measure import regionprops_table, regionprops, mesh_surface_area, marching_cubes, label
from skimage.morphology import binary_erosion
from skimage.measure._regionprops import _props_to_dict
from typing import Tuple, Any, Union

# from scipy.ndimage import maximum_position, center_of_mass
from scipy.ndimage import sum as ndi_sum
from scipy.sparse import coo_matrix

import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike

from infer_subc.core.img import apply_mask

import pandas as pd

def nrm_get_radial_distribution(
        cellmask_proj: np.ndarray,
        org_proj: np.ndarray,
        img_proj: np.ndarray,
        org_name: str,
        nucleus_proj: np.ndarray,
        n_bins: int = 5,
        from_edges: bool = True,
        keep_nuc_bins = True,
    ):
    """Perform the radial measurements on the image set

    Parameters
    ------------
    cellmask_proj: np.ndarray,
    org_proj: np.ndarray,
    img_proj: np.ndarray,
    org_name: str,
    nucleus_proj: np.ndarray,
    n_bins: int = 5,
    from_edges: bool = True,
    keep_nuc_bins = True,

    masked

    # params
    #   n_bins .e.g. 6
    #   normalizer - cellmask_voxels, organelle_voxels, cellmask_and_organelle_voxels
    #   from_edges = True


    Returns
    -------------
    returns one statistics table + bin_indexes image array
    """

    # other params
    bin_count = n_bins if n_bins is not None else 5
    nobjects = 1
    scale_bins = True
    ### I am removing this as a statement thus making it an option to omit or include the nucleus, default is true 
    # keep_nuc_bins = True
    center_on_nuc = False # choosing the edge of the nuclei or the center as the center to propogate from

    center_objects = nucleus_proj>0 

    # labels = label(cellmask_proj>0) #extent as 0,1 rather than bool

    # labels = np.zeros_like(cellmask_proj)
    # labels[labels>0]=1   

    labels = (cellmask_proj>0).astype(np.uint16)

    ### an if statement as part as the option to omit the nucleus 
    if not keep_nuc_bins:
        labels[nucleus_proj] = 0

    ################   ################
    ## define masks for computing distances
    ################   ################
    normalized_distance, good_mask, i_center, j_center = get_normalized_distance_and_mask(labels, center_objects, center_on_nuc, keep_nuc_bins)
    
    if normalized_distance is None:
        print('WTF!!  normailzed_distance returned wrong')

    ################   ################
    ## get histograms
    ################   ################
    ngood_pixels = np.sum(good_mask)
    good_labels = labels[good_mask]

    # protect against None normaized_distances
    if keep_nuc_bins:
        # For the exterior bins
        bin_indexes = ((normalized_distance * (bin_count - 1)) + 1).astype(int)
        bin_indexes[~good_mask] = 0
        # For the nucleus bin
        bin_indexes[center_objects] = 0
    else:
        bin_indexes = (normalized_distance * (bin_count)).astype(int)
    
    bin_indexes[bin_indexes > bin_count] = bin_count # shouldn't do anything

    #                 (    i          ,         j              )
    labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

    #                coo_matrix( (             data,             (i, j)    ), shape=                      )
    histogram_cmsk = coo_matrix( (cellmask_proj[good_mask], labels_and_bins), shape=(nobjects, bin_count) ).toarray()
    histogram_org = coo_matrix(  (org_proj[good_mask],      labels_and_bins), shape=(nobjects, bin_count) ).toarray()
    histogram_img = coo_matrix(  (img_proj[good_mask],      labels_and_bins), shape=(nobjects, bin_count) ).toarray()

    if keep_nuc_bins:
        # For the exterior bins
        bin_indexes = ((normalized_distance * (bin_count - 1)) + 1).astype(int)
        bin_indexes[~good_mask] = 0
        # For the nucleus bin
        bin_indexes[center_objects] = 0
    else:
        bin_indexes = (normalized_distance * (bin_count)).astype(int)

    sum_by_object_cmsk = np.sum(histogram_cmsk, 1) # flattened cellmask voxel count
    sum_by_object_org = np.sum(histogram_org, 1)  # organelle voxel count
    sum_by_object_img = np.sum(histogram_img, 1)  # image intensity projection

    # DEPRICATE: since we are NOT computing object_i by object_i (individual organelle labels)
    # sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]
    # fraction_at_distance = histogram / sum_by_object_per_bin

    # number of bins.
    number_at_distance = coo_matrix(( np.ones(ngood_pixels), labels_and_bins), (nobjects, bin_count)).toarray()

    # sicne we aren't breaking objects apart this is just ngood_pixels

    sum_by_object = np.sum(number_at_distance, 1)

    sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count))[0]
    fraction_at_bin = number_at_distance / sum_by_object_per_bin # sums to 1.0

    # object_mask = number_at_distance > 0
    # DEPRICATE:# not doing over multiple objects so don't need object mask.. or fractionals
    # mean_pixel_fraction = fraction_at_distance / ( fraction_at_bin + np.finfo(float).eps )
    # masked_fraction_at_distance = np.ma.masked_array( fraction_at_distance, ~object_mask )
    # masked_mean_pixel_fraction = np.ma.masked_array(mean_pixel_fraction, ~object_mask)

    ################   ################
    ## collect Anisotropy calculation.  + summarize
    ################   ################
    # Split each cell into eight wedges, then compute coefficient of variation of the wedges' mean intensities
    # in each ring. Compute each pixel's delta from the center object's centroid
    i, j = np.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
    imask = i[good_mask] > i_center[good_mask]
    jmask = j[good_mask] > j_center[good_mask]
    absmask = abs(i[good_mask] - i_center[good_mask]) > abs(
        j[good_mask] - j_center[good_mask]
    )
    radial_index = (
        imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4
    )

    # return radial_index, labels, good_mask, bin_indexes
    statistics = []
    stat_names =[]
    cv_cmsk = []
    cv_obj = []
    cv_img = []
    # collect the numbers from each "bin"
    for bin in range(bin_count):
        bin_mask = good_mask & (bin_indexes == bin)
        bin_pixels = np.sum(bin_mask)

        bin_labels = labels[bin_mask]

        bin_radial_index = radial_index[bin_indexes[good_mask] == bin]
        labels_and_radii = (bin_labels - 1, bin_radial_index)
        pixel_count = coo_matrix( (np.ones(bin_pixels), labels_and_radii), (nobjects, 8) ).toarray()

        radial_counts_cmsk = coo_matrix( (cellmask_proj[bin_mask], labels_and_radii), (nobjects, 8) ).toarray()
        radial_counts = coo_matrix( (org_proj[bin_mask], labels_and_radii), (nobjects, 8) ).toarray()
        radial_values = coo_matrix( (img_proj[bin_mask], labels_and_radii), (nobjects, 8) ).toarray()

        # we might need the masked arrays for some organelles... but I think not. keeping for now
        mask = pixel_count == 0

        radial_means_cmsk = np.ma.masked_array(radial_counts_cmsk / pixel_count, mask)
        radial_cv_cmsk = np.std(radial_means_cmsk, 1) / np.mean(radial_means_cmsk, 1)
        radial_cv_cmsk[np.sum(~mask, 1) == 0] = 0
        radial_cv_cmsk.mask = np.sum(~mask, 1) == 0


        radial_means_obj = np.ma.masked_array(radial_counts / pixel_count, mask)
        radial_cv_obj = np.std(radial_means_obj, 1) / np.mean(radial_means_obj, 1)
        radial_cv_obj[np.sum(~mask, 1) == 0] = 0
        radial_cv_obj.mask = np.sum(~mask, 1) == 0

        radial_means_img = np.ma.masked_array(radial_values / pixel_count, mask)
        radial_cv_img = np.std(radial_means_img, 1) / np.mean(radial_means_img, 1)
        radial_cv_img[np.sum(~mask, 1) == 0] = 0
        radial_cv_img.mask = np.sum(~mask, 1) == 0

        if keep_nuc_bins:
            bin_name = str(f'bin_{bin})') if bin > 0 else "nuc_bin"
        else: 
            bin_name = str(f'bin_{bin})')
        # # there's gotta be a better way to collect this stuff together... pandas?
        # statistics += [
        #     (   bin_name,
        #         # np.mean(number_at_distance[:, bin]), 
        #         # np.mean(histogram_cmsk[:, bin]), 
        #         # np.mean(histogram_org[:, bin]), 
        #         # np.mean(histogram_img[:, bin]), 
        #         np.mean(radial_cv_cmsk) ,
        #         np.mean(radial_cv_obj) ,
        #         np.mean(radial_cv_img) )
        # ]
        stat_names.append(bin_name)
        cv_cmsk.append(float(np.mean(radial_cv_cmsk)))  #convert to float to make importing from csv more straightforward
        cv_obj.append(float(np.mean(radial_cv_obj)))
        cv_img.append(float(np.mean(radial_cv_img))) # fixed the error

    # TODO: fix this grooooos hack
    # col_names=['organelle','mask','bin','n_bins','n_pix','cm_vox_cnt','org_vox_cnt','org_intensity','cm_radial_cv','org_radial_cv','img_radial_cv']
    # stats_dict={'organelle': org_name,
    #             'mask': 'cell',
    #             'radial_n_bins': bin_count,
    #             'radial_bins': [[s[0] for s in statistics]],
    #             'radial_cm_vox_cnt': [histogram_cmsk.squeeze().tolist()],
    #             'radial_org_vox_cnt': [histogram_org.squeeze().tolist()],
    #             'radial_org_intensity': [histogram_img.squeeze().tolist()],
    #             'radial_n_pix': [number_at_distance.squeeze().tolist()],
    #             'radial_cm_cv':[[s[1] for s in statistics]],
    #             'radial_org_cv':[[s[2] for s in statistics]],
    #             'radial_img_cv':[[s[3] for s in statistics]],
    #             }
    
    stats_dict={'organelle': org_name,
                'mask': 'cell',
                'radial_n_bins': bin_count,
                'radial_bins': [stat_names],
                'radial_cm_vox_cnt': [histogram_cmsk.squeeze().tolist()],
                'radial_org_vox_cnt': [histogram_org.squeeze().tolist()],
                'radial_org_intensity': [histogram_img.squeeze().tolist()],
                'radial_n_pix': [number_at_distance.squeeze().tolist()],
                'radial_cm_cv':[cv_cmsk],
                'radial_org_cv':[cv_obj],
                'radial_img_cv':[cv_img],
                }

    # stats_tab = pd.DataFrame(statistics,columns=col_names)
    stats_tab = pd.DataFrame(stats_dict)  
    return stats_tab, bin_indexes

def nrm_get_zernike_stats(        
        cellmask_proj: np.ndarray,
        org_proj:np.ndarray,
        img_proj: np.ndarray,
        organelle_name: str,
        nucleus_proj: Union[np.ndarray, None] = None,
        zernike_degree: int = 9):

    """
    
    """

    labels = label(cellmask_proj>0) #extent as 0,1 rather than bool
    zernike_indexes = centrosome.zernike.get_zernike_indexes( zernike_degree + 1)


    z = zernike_polynomial(labels, zernike_indexes)

    # included keep_nuc_bins as an option as that would change w

    z_cm = zernicke_stat(cellmask_proj, z)
    z_org = zernicke_stat(org_proj, z)
    z_nuc = zernicke_stat(nucleus_proj, z)
    z_img = zernicke_stat(img_proj, z)


    # nm_labels = [f"{n}_{m}" for (n, m) in (zernike_indexes)
    stats_tab = pd.DataFrame({'organelle':organelle_name,
                                'mask':'cell',
                                'zernike_n':[zernike_indexes[:,0].tolist()],
                                'zernike_m':[zernike_indexes[:,1].tolist()],
                                'zernike_cm_mag':[z_cm[0].tolist()],
                                'zernike_cm_phs':[z_cm[1].tolist()],   
                                'zernike_obj_mag':[z_org[0].tolist()],
                                'zernike_obj_phs':[z_org[1].tolist()],
                                'zernike_nuc_mag':[z_nuc[0].tolist()],
                                'zernike_nuc_phs':[z_nuc[1].tolist()],
                                'zernike_img_mag':[z_img[0].tolist()],
                                'zernike_img_phs':[z_img[1].tolist()]} # 'zernike_img_mag':[z_img[1].tolist()]} fixed error
                                )

    return stats_tab

def nrm_get_radial_stats(        
        cellmask_obj: np.ndarray,
        organelle_mask: np.ndarray,
        organelle_obj:np.ndarray,
        organelle_img: np.ndarray,
        organelle_name: str,
        nuclei_obj: np.ndarray,
        n_rad_bins: Union[int,None] = None,
        n_zernike: Union[int,None] = None,
        keep_nuc_bins: bool = True):

    """
    Params


    Returns
    -----------
    rstats table of radial distributions
    zstats table of zernike magnitudes and phases
    rad_bins image of the rstats bins over the cellmask_obj 

    """


    # flattened
    cellmask_proj = create_masked_Z_projection(cellmask_obj)
    org_proj = create_masked_Z_projection(organelle_obj,organelle_mask.astype(bool))
    # img_proj = create_masked_Z_projection(organelle_img,organelle_mask.astype(bool), to_bool=False)
    # It turns into a bool ndarray thus giving all intensities the same weight, which skew the data in favor of non organelle intensity
    img_proj = create_masked_Z_projection(organelle_img,organelle_mask, to_bool=False)

    nucleus_proj = create_masked_Z_projection(nuclei_obj,cellmask_obj.astype(bool)) 

    radial_stats, radial_bin_mask = nrm_get_radial_distribution(cellmask_proj=cellmask_proj, 
                                                            org_proj=org_proj, 
                                                            img_proj=img_proj, 
                                                            org_name=organelle_name, 
                                                            nucleus_proj=nucleus_proj, 
                                                            n_bins=n_rad_bins,
                                                            keep_nuc_bins=keep_nuc_bins)
    
    zernike_stats = nrm_get_zernike_stats(
                                      cellmask_proj=cellmask_proj, 
                                      org_proj=org_proj, 
                                      img_proj=img_proj, 
                                      organelle_name=organelle_name, 
                                      nucleus_proj=nucleus_proj, 
                                      zernike_degree = n_zernike)

    return radial_stats,zernike_stats,radial_bin_mask
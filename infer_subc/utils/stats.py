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

def _my_props_to_dict(
    rp, label_image, intensity_image=None, properties=("label", "area", "centroid", "bbox"), extra_properties=None
):
    """
    helper for get_summary_stats
    """
    if extra_properties is not None:
        properties = list(properties) + [prop.__name__ for prop in extra_properties]

    if len(rp) == 0:
        ndim = label_image.ndim
        label_image = np.zeros((3,) * ndim, dtype=int)
        label_image[(1,) * ndim] = 1
        if intensity_image is not None:
            intensity_image = np.zeros(label_image.shape + intensity_image.shape[ndim:], dtype=intensity_image.dtype)

        regions = regionprops(label_image, intensity_image=intensity_image, extra_properties=extra_properties)

        out_d = _props_to_dict(regions, properties=properties, separator="-")
        return {k: v[:0] for k, v in out_d.items()}

    return _props_to_dict(rp, properties=properties, separator="-")


def get_summary_stats_3D(input_labels: np.ndarray, intensity_img, mask: np.ndarray) -> Tuple[Any, Any]:
    """collect volumentric stats from skimage.measure.regionprops
        properties = ["label","max_intensity", "mean_intensity", "min_intensity" ,"area"->"volume" , "equivalent_diameter",
        "centroid", "bbox","euler_number", "extent"
        +   extra_properties = [standard_deviation_intensity]

    Parameters
    ------------
    input_labels:
        a 3d  np.ndarray image of the inferred organelle labels
    intensity_img:
        a 3d np.ndarray image of the florescence intensity
    mask:
        a 3d image containing the cellmask object (mask)

    Returns
    -------------
    pandas dataframe of stats and the regionprops object
    """

    # in case we sent a boolean mask (e.g. cyto, nucleus, cellmask)
    input_labels = _assert_uint16_labels(input_labels)

    # mask
    input_labels = apply_mask(input_labels, mask)

    # start with LABEL
    properties = ["label"]
    # add intensity:
    properties = properties + ["max_intensity", "mean_intensity", "min_intensity"]

    # arguments must be in the specified order, matching regionprops
    def standard_deviation_intensity(region, intensities):
        return np.std(intensities[region])

    extra_properties = [standard_deviation_intensity]

    # add area
    properties = properties + ["area", "equivalent_diameter"]
    #  position:
    properties = properties + ["centroid", "bbox"]  # , 'bbox', 'weighted_centroid']
    # etc
    properties = properties + ["euler_number", "extent"]  # only works for BIG organelles: 'convex_area','solidity',

    rp = regionprops(input_labels, intensity_image=intensity_img, extra_properties=extra_properties)

    props = _my_props_to_dict(
        rp, input_labels, intensity_image=intensity_img, properties=properties, extra_properties=extra_properties
    )

    props["surface_area"] = surface_area_from_props(input_labels, props)
    props_table = pd.DataFrame(props)
    props_table.rename(columns={"area": "volume"}, inplace=True)
    #  # ETC.  skeletonize via cellprofiler /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/morphologicalskeleton.py
    #         if x.volumetric:
    #             y_data = skimage.morphology.skeletonize_3d(x_data)
    # /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/measureobjectskeleton.py

    return props_table, rp


def surface_area_from_props(labels, props):
    """helper function for getting surface area of volumetric segmentation"""

    # SurfaceArea
    surface_areas = np.zeros(len(props["label"]))
    # TODO: spacing = [1, 1, 1] # this is where we could deal with anisotropy in Z

    for index, lab in enumerate(props["label"]):
        # this seems less elegant than you might wish, given that regionprops returns a slice,
        # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
        volume = labels[
            max(props["bbox-0"][index] - 1, 0) : min(props["bbox-3"][index] + 1, labels.shape[0]),
            max(props["bbox-1"][index] - 1, 0) : min(props["bbox-4"][index] + 1, labels.shape[1]),
            max(props["bbox-2"][index] - 1, 0) : min(props["bbox-5"][index] + 1, labels.shape[2]),
        ]
        volume = volume == lab
        verts, faces, _normals, _values = marching_cubes(
            volume,
            method="lewiner",
            spacing=(1.0,) * labels.ndim,
            level=0,
        )
        surface_areas[index] = mesh_surface_area(verts, faces)

    return surface_areas


def _assert_uint16_labels(inp: np.ndarray) -> np.ndarray:
    """
    wrapper to enforce having the right labels
    """
    if inp.dtype == "bool" or inp.dtype == np.uint8:
        return label(inp > 0).astype(np.uint16)
    return inp


def get_aXb_stats_3D(a, b, mask, use_shell_a=False):
    """
    collect volumentric stats of `a` intersect `b`
    """
    properties = ["label"]  # our index to organelles
    # add area
    properties = properties + ["area", "equivalent_diameter"]
    #  position:
    properties = properties + ["centroid", "bbox"]  # ,  'weighted_centroid']
    # etc
    properties = properties + ["slice"]

    # in case we sent a boolean mask (e.g. cyto, nucleus, cellmask)
    a = _assert_uint16_labels(a)
    b = _assert_uint16_labels(b)

    if use_shell_a:
        a_int_b = np.logical_and(np.logical_xor(a > 0, binary_erosion(a > 0)), b > 0)
    else:
        a_int_b = np.logical_and(a > 0, b > 0)

    labels = label(apply_mask(a_int_b, mask)).astype("int")

    props = regionprops_table(labels, intensity_image=None, properties=properties, extra_properties=None)

    props["surface_area"] = surface_area_from_props(labels, props)
    # there could be a bug here if there are spurious labels in the corners of the slices

    label_a = []
    index_ab = []
    label_b = []
    for index, lab in enumerate(props["label"]):
        # this seems less elegant than you might wish, given that regionprops returns a slice,
        # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
        volume = labels[props["slice"][index]]
        la = a[props["slice"][index]]
        lb = b[props["slice"][index]]
        volume = volume == lab
        la = la[volume]
        lb = lb[volume]

        all_as = np.unique(la[la>0]).tolist()
        all_bs = np.unique(lb[lb>0]).tolist()
        if len(all_as) != 1:
            print(f"we have an error.  as-> {all_as}")
        if len(all_bs) != 1:
            print(f"we have an error.  bs-> {all_bs}")

        label_a.append(all_as[0] )
        label_b.append(all_bs[0] )
        index_ab.append(f"{all_as[0]}_{all_bs[0]}") 


    props["label_a"] = label_a #[np.unique(a[s])[:1].tolist() for s in props["slice"]]
    props["label_b"] = label_b #[np.unique(b[s])[:1].tolist() for s in props["slice"]]
    props_table = pd.DataFrame(props)
    props_table.rename(columns={"area": "volume"}, inplace=True)
    props_table.drop(columns="slice", inplace=True)
    props_table.insert(loc=0,column='label_',value=index_ab)
    props_table.insert(loc=0,column='shell',value=use_shell_a)

    return props_table


def create_masked_Z_projection(img_in:np.ndarray, mask:Union[np.ndarray, None]=None, to_bool:bool=True) -> np.ndarray:
    """
    
    """
    img_out = img_in.astype(bool) if to_bool else img_in
    if mask is not None:
        img_out = apply_mask(img_out, mask)
    
    return img_out.sum(axis=0)



###################################
### DISTRIBUTIONAL STATS
###################################


# taken from cellprofiler_core.utilities.core.object
def size_similarly(labels, secondary):
    """Size the secondary matrix similarly to the labels matrix

    labels - labels matrix
    secondary - a secondary image or labels matrix which might be of
                different size.
    Return the resized secondary matrix and a mask indicating what portion
    of the secondary matrix is bogus (manufactured values).

    Either the mask is all ones or the result is a copy, so you can
    modify the output within the unmasked region w/o destroying the original.
    """
    if labels.shape[:2] == secondary.shape[:2]:
        return secondary, np.ones(secondary.shape, bool)
    if labels.shape[0] <= secondary.shape[0] and labels.shape[1] <= secondary.shape[1]:
        if secondary.ndim == 2:
            return (
                secondary[: labels.shape[0], : labels.shape[1]],
                np.ones(labels.shape, bool),
            )
        else:
            return (
                secondary[: labels.shape[0], : labels.shape[1], :],
                np.ones(labels.shape, bool),
            )

    # Some portion of the secondary matrix does not cover the labels
    result = np.zeros(
        list(labels.shape) + list(secondary.shape[2:]), secondary.dtype
    )
    i_max = min(secondary.shape[0], labels.shape[0])
    j_max = min(secondary.shape[1], labels.shape[1])
    if secondary.ndim == 2:
        result[:i_max, :j_max] = secondary[:i_max, :j_max]
    else:
        result[:i_max, :j_max, :] = secondary[:i_max, :j_max, :]
    mask = np.zeros(labels.shape, bool)
    mask[:i_max, :j_max] = 1
    return result, mask




def get_radial_stats(        
        cellmask_obj: np.ndarray,
        organelle_mask: np.ndarray,
        organelle_obj:np.ndarray,
        organelle_img: np.ndarray,
        organelle_name: str,
        nuclei_obj: np.ndarray,
        n_rad_bins: Union[int,None] = None,
        n_zernike: Union[int,None] = None,
        ):

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
    img_proj = create_masked_Z_projection(organelle_img,organelle_mask.astype(bool), to_bool=False)

    nucleus_proj = create_masked_Z_projection(nuclei_obj,cellmask_obj.astype(bool)) 

    radial_stats, radial_bin_mask = get_radial_distribution(cellmask_proj=cellmask_proj, 
                                                            org_proj=org_proj, 
                                                            img_proj=img_proj, 
                                                            org_name=organelle_name, 
                                                            nucleus_proj=nucleus_proj, 
                                                            n_bins=n_rad_bins
                                                            )
    
    zernike_stats = get_zernike_stats(
                                      cellmask_proj=cellmask_proj, 
                                      org_proj=org_proj, 
                                      img_proj=img_proj, 
                                      organelle_name=organelle_name, 
                                      nucleus_proj=nucleus_proj, 
                                      zernike_degree = n_zernike
                                      )

    return radial_stats,zernike_stats,radial_bin_mask
    
def get_normalized_distance_and_mask(labels, center_objects, center_on_nuc, keep_nuc_bins):
    """
    helper for radial distribution
    """
    d_to_edge = centrosome.cpmorphology.distance_to_edge(labels) # made a local version
    ## use the nucleus as the center 
    if center_objects is not None:
        # don't need to do this size_similarity trick.  I KNOW that labels and center_objects are the same size
        # center_labels, cmask = size_similarly(labels, center_objects)
        # going to leave them as labels, so in princi    ple the same code could work for partitioned masks (labels)
        center_labels = label(center_objects)
        pixel_counts = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                            ndi_sum(
                                np.ones(center_labels.shape),
                                center_labels,
                                np.arange(
                                    1, np.max(center_labels) + 1, dtype=np.int32
                                ),
                            )
                        )
        good = pixel_counts > 0
        i, j = ( centrosome.cpmorphology.centers_of_labels(center_labels) + 0.5).astype(int)
        ig = i[good]
        jg = j[good]
        lg = np.arange(1, len(i) + 1)[good]
        
        if center_on_nuc:  # Reduce the propagation labels to the centers of the centering objects
            center_labels = np.zeros(center_labels.shape, int)
            center_labels[ig, jg] = lg


        cl, d_from_center = centrosome.propagate.propagate(  np.zeros(center_labels.shape), center_labels, labels != 0, 1)
        cl[labels == 0] = 0            # Erase the centers that fall outside of labels


        # SHOULD NEVER NEED THIS because we arent' looking at multiple
        # If objects are hollow or crescent-shaped, there may be objects without center labels. As a backup, find the
        # center that is the closest to the center of mass.
        missing_mask = (labels != 0) & (cl == 0)
        missing_labels = np.unique(labels[missing_mask])
        
        if len(missing_labels):
            print("WTF!!  how did we have missing labels?")
            all_centers = centrosome.cpmorphology.centers_of_labels(labels)
            missing_i_centers, missing_j_centers = all_centers[:, missing_labels-1]
            di = missing_i_centers[:, np.newaxis] - ig[np.newaxis, :]
            dj = missing_j_centers[:, np.newaxis] - jg[np.newaxis, :]
            missing_best = lg[np.argsort(di * di + dj * dj)[:, 0]]
            best = np.zeros(np.max(labels) + 1, int)
            best[missing_labels] = missing_best
            cl[missing_mask] = best[labels[missing_mask]]

            # Now compute the crow-flies distance to the centers of these pixels from whatever center was assigned to the object.
            iii, jjj = np.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
            di = iii[missing_mask] - i[cl[missing_mask] - 1]
            dj = jjj[missing_mask] - j[cl[missing_mask] - 1]
            d_from_center[missing_mask] = np.sqrt(di * di + dj * dj)

        good_mask = cl > 0

        if not keep_nuc_bins:
            # include pixels within the centering objects
            # when performing calculations from the centers
            good_mask = good_mask & (center_labels == 0)
            
    else: # ELSE     if center_objects is  None so center on the middle of the cellmask_mask
        i, j = centrosome.cpmorphology.maximum_position_of_labels(   d_to_edge, labels, [1])
        center_labels = np.zeros(labels.shape, int)
        center_labels[i, j] = labels[i, j]
        # Use the coloring trick here to process touching objectsin separate operations
        colors = centrosome.cpmorphology.color_labels(labels)
        ncolors = np.max(colors)
        d_from_center = np.zeros(labels.shape)
        cl = np.zeros(labels.shape, int)

        for color in range(1, ncolors + 1):
            mask = colors == color
            l, d = centrosome.propagate.propagate( np.zeros(center_labels.shape), center_labels, mask, 1)
            d_from_center[mask] = d[mask]
            cl[mask] = l[mask]

        good_mask = cl > 0

    ## define spatial distribution from masks
    # collect arrays of centers
    i_center = np.zeros(cl.shape)
    i_center[good_mask] = i[cl[good_mask] - 1]
    j_center = np.zeros(cl.shape)
    j_center[good_mask] = j[cl[good_mask] - 1]

    normalized_distance = np.zeros(labels.shape)
    total_distance = d_from_center + d_to_edge

    normalized_distance[good_mask] = d_from_center[good_mask] / ( total_distance[good_mask] + 0.001 )
    return normalized_distance, good_mask, i_center, j_center


def get_radial_distribution(
        cellmask_proj: np.ndarray,
        org_proj: np.ndarray,
        img_proj: np.ndarray,
        org_name: str,
        nucleus_proj: np.ndarray,
        n_bins: int = 5,
        from_edges: bool = True,
    ):
    """Perform the radial measurements on the image set

    Parameters
    ------------
    cellmask_proj: np.ndarray,
    org_proj: np.ndarray,
    img_proj: np.ndarray,
    org_name: str,
    nucleus_proj: Union[np.ndarray, None],
    n_bins: int = 5,
    from_edges: bool = True,

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
    keep_nuc_bins = True # this toggles whether to count things inside the nuclei mask.  
    center_on_nuc = False # choosing the edge of the nuclei or the center as the center to propogate from

    center_objects = nucleus_proj>0 

    # labels = label(cellmask_proj>0) #extent as 0,1 rather than bool    
    labels = (cellmask_proj>0).astype(np.uint16)
    # labels = np.zeros_like(cellmask_proj)
    # labels[labels>0]=1

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
    bin_indexes = (normalized_distance * bin_count).astype(int)
    bin_indexes[bin_indexes > bin_count] = bin_count # shouldn't do anything

    #                 (    i          ,         j              )
    labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

    #                coo_matrix( (             data,             (i, j)    ), shape=                      )
    histogram_cmsk = coo_matrix( (cellmask_proj[good_mask], labels_and_bins), shape=(nobjects, bin_count) ).toarray()
    histogram_org = coo_matrix(  (org_proj[good_mask],      labels_and_bins), shape=(nobjects, bin_count) ).toarray()
    histogram_img = coo_matrix(  (img_proj[good_mask],      labels_and_bins), shape=(nobjects, bin_count) ).toarray()

    bin_indexes = (normalized_distance * bin_count).astype(int)

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

        bin_name = str(bin) if bin > 0 else "Ctr"

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
        cv_img.append(float(np.mean(radial_cv_obj)))

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


def create_masked_depth_projection(img_in:np.ndarray, mask:Union[np.ndarray, None]=None, to_bool:bool=True) -> np.ndarray:
    """
    create masked projection onto the Z dimension
    """
    img_out = img_in.astype(bool) if to_bool else img_in
    if mask is not None:
        img_out = apply_mask(img_out, mask)
    
    return img_out.sum(axis=(1,2))


def get_depth_stats(        
        cellmask_obj: np.ndarray,
        organelle_mask: np.ndarray,
        organelle_obj:np.ndarray,
        organelle_img: np.ndarray,
        organelle_name: str,
        nuclei_obj: Union[np.ndarray, None],
        ):
    """

    """

    # flattened
    cellmask_proj = create_masked_depth_projection(cellmask_obj)
    org_proj = create_masked_depth_projection(organelle_obj,organelle_mask.astype(bool))
    img_proj = create_masked_depth_projection(organelle_img,organelle_mask.astype(bool), to_bool=False)

    nucleus_proj = create_masked_depth_projection(nuclei_obj,cellmask_obj.astype(bool)) if nuclei_obj is not None else None
    z_bins = [i for i in range(cellmask_obj.shape[0])]

    stats_tab = pd.DataFrame({'organelle':organelle_name,
                            'mask':'cell',
                            'n_z':cellmask_obj.shape[0],
                            'z':[z_bins],
                            'z_cm_vox_cnt':[cellmask_proj.tolist()],
                            'z_org_vox_cnt':[org_proj.tolist()],
                            'z_org_intensity':[img_proj.tolist()],
                            'z_nuc_vox_cnt':[nucleus_proj.tolist()]})
    return stats_tab
    

# Zernicke routines.  inspired by cellprofiler, but heavily simplified
def zernicke_stat(pixels,z):
    """
    
    """
    vr = np.sum(pixels[:,:,np.newaxis]*z.real, axis=(0,1))
    vi = np.sum(pixels[:,:,np.newaxis]*z.imag, axis=(0,1))    
    magnitude = np.sqrt(vr * vr + vi * vi) / pixels.sum()
    phase = np.arctan2(vr, vi)
    # return {"zer_mag": magnitude, "zer_phs": phase}
    return magnitude, phase



def zernike_polynomial(labels, zernike_is):
    """
    

    """
    # First, get a table of centers and radii of minimum enclosing
    # circles for the cellmask
    ij, r = centrosome.cpmorphology.minimum_enclosing_circle( labels )
    # Then compute x and y, the position of each labeled pixel
    # within a unit circle around the object
    iii, jjj = np.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]

    # translate+scale
    iii = (iii-ij[0][0] ) / r
    jjj = (jjj-ij[0][1] ) / r

    z = centrosome.zernike.construct_zernike_polynomials(
        iii, jjj, zernike_is
    )
    return z
    

   
def get_zernike_stats(        
        cellmask_proj: np.ndarray,
        org_proj:np.ndarray,
        img_proj: np.ndarray,
        organelle_name: str,
        nucleus_proj: Union[np.ndarray, None] = None,
        zernike_degree: int = 9
        ):

    """
    
    """

    labels = label(cellmask_proj>0) #extent as 0,1 rather than bool
    zernike_indexes = centrosome.zernike.get_zernike_indexes( zernike_degree + 1)


    z = zernike_polynomial(labels, zernike_indexes)

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
                                'zernike_img_mag':[z_img[1].tolist()]}
                                )

    return stats_tab


# DEPRICATE
def get_simple_stats_3D(a, mask):
    """collect volumentric stats of `a`"""

    properties = ["label"]  # our index to organelles
    # add area
    properties = properties + ["area", "equivalent_diameter"]
    #  position:
    properties = properties + ["centroid", "bbox"]  # ,  'weighted_centroid']
    # etc
    properties = properties + ["slice"]

    # in case we sent a boolean mask (e.g. cyto, nucleus, cellmask)
    labels = _assert_uint16_labels(a)

    # props = regionprops_table(labels, intensity_image=None,
    #                             properties=properties, extra_properties=[])

    rp = regionprops(labels, intensity_image=None, extra_properties=[])
    props = _my_props_to_dict(rp, labels, intensity_image=None, properties=properties, extra_properties=None)

    stats_table = pd.DataFrame(props)
    stats_table.rename(columns={"area": "volume"}, inplace=True)

    return stats_table, rp


# untested 2D version
def get_summary_stats_2D(input_labels, intensity_img, mask):
    """collect volumentric stats"""

    # in case we sent a boolean mask (e.g. cyto, nucleus, cellmask)
    input_labels = _assert_uint16_labels(input_labels)

    # mask
    input_labels = apply_mask(input_labels, mask)

    properties = ["label"]
    # add intensity:
    properties = properties + ["max_intensity", "mean_intensity", "min_intensity"]

    # arguments must be in the specified order, matching regionprops
    def standard_deviation_intensity(region, intensities):
        return np.std(intensities[region])

    extra_properties = [standard_deviation_intensity]

    # add area
    properties = properties + ["area", "equivalent_diameter"]
    #  position:
    properties = properties + ["centroid", "bbox"]

    #  perimeter:
    properties = properties + ["perimeter", "perimeter_crofton"]

    rp = regionprops(input_labels, intensity_image=intensity_img, extra_properties=extra_properties)
    props = _my_props_to_dict(
        rp, input_labels, intensity_image=intensity_img, properties=properties, extra_properties=extra_properties
    )
    props_table = pd.DataFrame(props)
    #  # ETC.  skeletonize via cellprofiler /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/morphologicalskeleton.py
    #             y_data = skimage.morphology.skeletonize(x_data)
    # /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/measureobjectskeleton.py
    return props_table, rp

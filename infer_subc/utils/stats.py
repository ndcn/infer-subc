# from typing import List
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, regionprops, mesh_surface_area, marching_cubes, label
# from skimage.morphology import binary_erosion
from skimage.measure._regionprops import _props_to_dict
from typing import Tuple, Any, Union
import itertools

# from scipy.ndimage import maximum_position, center_of_mass
from scipy.ndimage import sum as ndi_sum
from scipy.sparse import coo_matrix

import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike

from infer_subc.core.img import apply_mask



def _my_props_to_dict(
    rp, label_image, intensity_image=None, properties=("label", "area", "centroid", "bbox"), extra_properties=None, spacing: Union[tuple, None] = None):
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

        regions = regionprops(label_image, intensity_image=intensity_image, extra_properties=extra_properties, spacing=spacing)

        out_d = _props_to_dict(regions, properties=properties, separator="-")
        return {k: v[:0] for k, v in out_d.items()}

    return _props_to_dict(rp, properties=properties, separator="-")

### USED ###
def get_org_morphology_3D(segmentation_img: np.ndarray, 
                           seg_name: str, 
                           intensity_img, 
                           mask: np.ndarray, 
                           scale: Union[tuple, None]=None):
    """
    Parameters
    ------------
    segmentation_img:
        a 3D (ZYX) np.ndarray of segmented objects 
    seg_name: str
        a name or nickname of the object being measured; this will be used for record keeping in the output table
    intensity_img:
        a 3D (ZYX) np.ndarray contain gray scale values from the "raw" image the segmentation is based on )single channel)
    mask:
        a 3D (ZYX) binary np.ndarray mask of the area to measure from
    scale: tuple, optional
        a tuple that contains the real world dimensions for each dimension in the image (Z, Y, X)


    Regionprops measurements:
    ------------------------
    ['label',
    'centroid',
    'bbox',
    'area',
    'equivalent_diameter',
    'extent',
    'feret_diameter_max',
    'euler_number',
    'convex_area',
    'solidity',
    'axis_major_length',
    'axis_minor_length',
    'max_intensity',
    'mean_intensity',
    'min_intensity']

    Additional measurements:
    -----------------------
    ['standard_deviation_intensity',
    'surface_area']


    Returns
    -------------
    pandas dataframe of containing regionprops measurements (columns) for each object in the segmentation image (rows) and the regionprops object
    
    """
    ###################################################
    ## MASK THE ORGANELLE OBJECTS THAT WILL BE MEASURED
    ###################################################
    # in case we sent a boolean mask (e.g. cyto, nucleus, cellmask)
    input_labels = _assert_uint16_labels(segmentation_img)

    # mask
    input_labels = apply_mask(input_labels, mask)

    ##########################################
    ## CREATE LIST OF REGIONPROPS MEASUREMENTS
    ##########################################
    # start with LABEL
    properties = ["label"]

    # add position
    properties = properties + ["centroid", "bbox"]

    # add area
    properties = properties + ["area", "equivalent_diameter"] # "num_pixels", 

    # add shape measurements
    properties = properties + ["extent", "euler_number", "solidity", "axis_major_length"] # ,"feret_diameter_max", "axis_minor_length"]

    # add intensity values (used for quality checks)
    properties = properties + ["min_intensity", "max_intensity", "mean_intensity"]

    #######################
    ## ADD EXTRA PROPERTIES
    #######################
    def standard_deviation_intensity(region, intensities):
        return np.std(intensities[region])

    extra_properties = [standard_deviation_intensity]

    ##################
    ## RUN REGIONPROPS
    ##################
    props = regionprops_table(input_labels, 
                           intensity_image=intensity_img, 
                           properties=properties,
                           extra_properties=extra_properties,
                           spacing=scale)

    props_table = pd.DataFrame(props)
    props_table.insert(0, "object", seg_name)
    props_table.rename(columns={"area": "volume"}, inplace=True)

    if scale is not None:
        round_scale = (round(scale[0], 4), round(scale[1], 4), round(scale[2], 4))
        props_table.insert(loc=2, column="scale", value=f"{round_scale}")
    else: 
        props_table.insert(loc=2, column="scale", value=f"{tuple(np.ones(segmentation_img.ndim))}") 

    ##################################################################
    ## RUN SURFACE AREA FUNCTION SEPARATELY AND APPEND THE PROPS_TABLE
    ##################################################################
    surface_area_tab = pd.DataFrame(surface_area_from_props(input_labels, props, scale))

    props_table.insert(12, "surface_area", surface_area_tab)
    props_table.insert(14, "SA_to_volume_ratio", props_table["surface_area"].div(props_table["volume"]))

    ################################################################
    ## ADD SKELETONIZATION OPTION FOR MEASURING LENGTH AND BRANCHING
    ################################################################


    return props_table


# def get_summary_stats_3D(input_labels: np.ndarray, intensity_img, mask: np.ndarray) -> Tuple[Any, Any]:
#     """collect volumentric stats from skimage.measure.regionprops
#         properties = ["label","max_intensity", "mean_intensity", "min_intensity" ,"area"->"volume" , "equivalent_diameter",
#         "centroid", "bbox","euler_number", "extent"
#         +   extra_properties = [standard_deviation_intensity]

#     Parameters
#     ------------
#     input_labels:
#         a 3d  np.ndarray image of the inferred organelle labels
#     intensity_img:
#         a 3d np.ndarray image of the florescence intensity
#     mask:
#         a 3d image containing the cellmask object (mask)

#     Returns
#     -------------
#     pandas dataframe of stats and the regionprops object
#     """

#     # in case we sent a boolean mask (e.g. cyto, nucleus, cellmask)
#     input_labels = _assert_uint16_labels(input_labels)

#     # mask
#     input_labels = apply_mask(input_labels, mask)

#     # start with LABEL
#     properties = ["label"]
#     # add intensity:
#     properties = properties + ["max_intensity", "mean_intensity", "min_intensity"]

#     # arguments must be in the specified order, matching regionprops
#     def standard_deviation_intensity(region, intensities):
#         return np.std(intensities[region])

#     extra_properties = [standard_deviation_intensity]

#     # add area
#     properties = properties + ["area", "equivalent_diameter"]
#     #  position:
#     properties = properties + ["centroid", "bbox"]  # , 'bbox', 'weighted_centroid']
#     # etc
#     properties = properties + ["euler_number", "extent"]  # only works for BIG organelles: 'convex_area','solidity',

#     rp = regionprops(input_labels, intensity_image=intensity_img, extra_properties=extra_properties)

#     props = _my_props_to_dict(
#         rp, input_labels, intensity_image=intensity_img, properties=properties, extra_properties=extra_properties
#     )

#     props["surface_area"] = surface_area_from_props(input_labels, props)
#     props_table = pd.DataFrame(props)
#     props_table.rename(columns={"area": "volume"}, inplace=True)
#     #  # ETC.  skeletonize via cellprofiler /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/morphologicalskeleton.py
#     #         if x.volumetric:
#     #             y_data = skimage.morphology.skeletonize_3d(x_data)
#     # /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/measureobjectskeleton.py

#     return props_table, rp


# creating a function to measure the surface area of each object. This function utilizes "marching_cubes" to generate a mesh (non-pixelated object)

### USED ###
def surface_area_from_props(labels, props, scale: Union[tuple,None]=None):
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
        if scale is None:
            scale=(1.0,) * labels.ndim
        verts, faces, _normals, _values = marching_cubes(
            volume,
            method="lewiner",
            spacing=scale,
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

### USED ###
def get_region_morphology_3D(region_seg: np.ndarray, 
                              region_name: str,
                              intensity_img: np.ndarray, 
                              channel_names: [str],
                              mask: np.ndarray, 
                              scale: Union[tuple, None]=None) -> Tuple[Any, Any]:
    """
    Parameters
    ------------
    region_seg:
        a 3D (ZYX) np.ndarray of segmented objects 
    region_name: str
        a name or nickname of the object being measured; this will be used for record keeping in the output table
    intensity_img:
        a 3D (ZYX) np.ndarray contain gray scale values from the "raw" image the segmentation is based on )single channel)
    mask:
        a 3D (ZYX) binary np.ndarray mask of the area to measure from
    scale: tuple, optional
        a tuple that contains the real world dimensions for each dimension in the image (Z, Y, X)


    Regionprops measurements:
    ------------------------
    ['label',
    'centroid',
    'bbox',
    'area',
    'equivalent_diameter',
    'extent',
    'feret_diameter_max',
    'euler_number',
    'convex_area',
    'solidity',
    'axis_major_length',
    'axis_minor_length',
    'max_intensity',
    'mean_intensity',
    'min_intensity']

    Additional measurements:
    -----------------------
    ['standard_deviation_intensity',
    'surface_area']


    Returns
    -------------
    pandas dataframe of containing regionprops measurements (columns) for each object in the segmentation image (rows) and the regionprops object

    """
    if len(channel_names) != intensity_img.shape[0]:
        ValueError("You have not provided a name for each channel in the intensity image. Make sure there is a channel name for each channel in the intensity image.")
    
    ###################################################
    ## MASK THE REGION OBJECTS THAT WILL BE MEASURED
    ###################################################
    # in case we sent a boolean mask (e.g. cyto, nucleus, cellmask)
    input_labels = _assert_uint16_labels(region_seg)

    input_labels = apply_mask(input_labels, mask)

    ##########################################
    ## CREATE LIST OF REGIONPROPS MEASUREMENTS
    ##########################################
    # start with LABEL
    properties = ["label"]
    # add position
    properties = properties + ["centroid", "bbox"]
    # add area
    properties = properties + ["area", "equivalent_diameter"] # "num_pixels", 
    # add shape measurements
    properties = properties + ["extent", "euler_number", "solidity", "axis_major_length"] # ,"feret_diameter_max", , "axis_minor_length"]
    # add intensity values (used for quality checks)
    properties = properties + ["min_intensity", "max_intensity", "mean_intensity"]

    #######################
    ## ADD EXTRA PROPERTIES
    #######################
    def standard_deviation_intensity(region, intensities):
        return np.std(intensities[region])

    extra_properties = [standard_deviation_intensity]

    ##################
    ## RUN REGIONPROPS
    ##################
    intensity_input = np.moveaxis(intensity_img, 0, -1)

    rp = regionprops(input_labels, 
                    intensity_image=intensity_input, 
                    extra_properties=extra_properties, 
                    spacing=scale)

    props = regionprops_table(label_image=input_labels, 
                              intensity_image=intensity_input, 
                              properties=properties, 
                              extra_properties=extra_properties,
                              spacing=scale)

    props_table = pd.DataFrame(props)
    props_table.insert(0, "object", region_name)
    props_table.rename(columns={"area": "volume"}, inplace=True)

    if scale is not None:
        round_scale = (round(scale[0], 4), round(scale[1], 4), round(scale[2], 4))
        props_table.insert(loc=2, column="scale", value=f"{round_scale}")
    else: 
        props_table.insert(loc=2, column="scale", value=f"{tuple(np.ones(region_seg.ndim))}") 

    rename_dict = {}
    for col in props_table.columns:
        for idx, name in enumerate(channel_names):
            if col.endswith(f"intensity-{idx}"):
                rename_dict[f"{col}"] = f"{col[:-1]}{name}_ch"

    props_table = props_table.rename(columns=rename_dict)

    ##################################################################
    ## RUN SURFACE AREA FUNCTION SEPARATELY AND APPEND THE PROPS_TABLE
    ##################################################################
    surface_area_tab = pd.DataFrame(surface_area_from_props(input_labels, props, scale))

    props_table.insert(12, "surface_area", surface_area_tab)
    props_table.insert(14, "SA_to_volume_ratio", props_table["surface_area"].div(props_table["volume"]))

    ################################################################
    ## ADD SKELETONIZATION OPTION FOR MEASURING LENGTH AND BRANCHING
    ################################################################
    #  # ETC.  skeletonize via cellprofiler /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/morphologicalskeleton.py
    #         if x.volumetric:
    #             y_data = skimage.morphology.skeletonize_3d(x_data)
    # /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/measureobjectskeleton.py

    return props_table

### USED ###
def get_contact_metrics_3D(a: np.ndarray,
                            a_name: str, 
                            b: np.ndarray, 
                            b_name:str, 
                            mask: np.ndarray, 
                            scale: Union[tuple, None]=None,
                            include_dist:bool=False, 
                            dist_centering_obj: Union[np.ndarray, None]=None,
                            dist_num_bins: Union[int, None]=None,
                            dist_zernike_degrees: Union[int, None]=None,
                            dist_center_on: Union[bool, None]=None,
                            dist_keep_center_as_bin: Union[bool, None]=None):
    """
    collect volumentric measurements of organelle `a` intersect organelle `b`

    Parameters
    ------------
    a: np.ndarray
        3D (ZYX) np.ndarray of one set of objects that will be assessed as part of a "contact"
    a_name: str
        the name or nickname of object a; this will be used for record keeping purposed in the output dataframe 
    b: np.ndarray
        3D (ZYX) np.ndarray of one set of objects that will be assessed as part of a "contact"
    b_name: str
        the name or nickname of object b; this will be used for record keeping purposed in the output dataframe 
    mask: np.ndarray
        3D (ZYX) binary mask of the area to measure contacts from
    include_dist:bool=False
        *optional*
        True = include the XY and Z distribution measurements of the contact sites within the masked region 
        (utilizing the functions get_XY_distribution() and get_Z_distribution() from Infer-subc)
        False = do not include distirbution measurements
    dist_centering_obj: Union[np.ndarray, None]=None
        ONLY NEEDED IF include_dist=True; if None, the center of the mask will be used
        3D (ZYX) np.ndarray containing the object to use for centering the XY distribution mask
    dist_num_bins: Union[int, None]=None
        ONLY NEEDED IF include_dist=True; if None, the default is 5
    dist_zernike_degrees: Unions[int, None]=None,
        ONLY NEEDED IF include_dist=True; if None, the zernike share measurements will not be included in the distribution
        the number of zernike degrees to include for the zernike shape descriptors
    dist_center_on: Union[bool, None]=None
        ONLY NEEDED IF include_dist=True; if None, the default is False
        True = distribute the bins from the center of the centering object
        False = distribute the bins from the edge of the centering object
    dist_keep_center_as_bin: Union[bool, None]=None
        ONLY NEEDED IF include_dist=True; if None, the default is True
        True = include the centering object area when creating the bins
        False = do not include the centering object area when creating the bins


    Regionprops measurements:
    ------------------------
    ['label',
    'centroid',
    'bbox',
    'area',
    'equivalent_diameter',
    'extent',
    'feret_diameter_max',
    'euler_number',
    'convex_area',
    'solidity',
    'axis_major_length',
    'axis_minor_length']

    Additional measurements:
    ----------------------
    ['surface_area']

    
    Returns
    -------------
    pandas dataframe of containing regionprops measurements (columns) for each overlap region between a and b (rows)
    
    """
    
    #########################
    ## CREATE OVERLAP REGIONS
    #########################
    a = _assert_uint16_labels(a)
    b = _assert_uint16_labels(b)

    a_int_b = np.logical_and(a > 0, b > 0)

    labels = label(apply_mask(a_int_b, mask)).astype("int")

    ##########################################
    ## CREATE LIST OF REGIONPROPS MEASUREMENTS
    ##########################################
    # start with LABEL
    properties = ["label"]

    # add position
    properties = properties + ["centroid", "bbox"]

    # add area
    properties = properties + ["area", "equivalent_diameter"] # "num_pixels", 

    # add shape measurements
    properties = properties + ["extent", "euler_number", "solidity", "axis_major_length", "slice"] # "feret_diameter_max", "axis_minor_length", 

    ##################
    ## RUN REGIONPROPS
    ##################
    props = regionprops_table(labels, intensity_image=None, properties=properties, extra_properties=None, spacing=scale)

    ##################################################################
    ## RUN SURFACE AREA FUNCTION SEPARATELY AND APPEND THE PROPS_TABLE
    ##################################################################
    surface_area_tab = pd.DataFrame(surface_area_from_props(labels, props, scale))

    ######################################################
    ## LIST WHICH ORGANELLES ARE INVOLVED IN THE CONTACTS
    ######################################################
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

        label_a.append(f"{all_as[0]}" )
        label_b.append(f"{all_bs[0]}" )
        index_ab.append(f"{all_as[0]}_{all_bs[0]}")


    ######################################################
    ## CREATE COMBINED DATAFRAME OF THE QUANTIFICATION
    ######################################################
    props_table = pd.DataFrame(props)
    props_table.drop(columns=['slice', 'label'], inplace=True)
    props_table.insert(0, 'label',value=index_ab)
    props_table.insert(0, "object", f"{a_name}X{b_name}")
    props_table.rename(columns={"area": "volume"}, inplace=True)

    props_table.insert(11, "surface_area", surface_area_tab)
    props_table.insert(13, "SA_to_volume_ratio", props_table["surface_area"].div(props_table["volume"]))

    if scale is not None:
        round_scale = (round(scale[0], 4), round(scale[1], 4), round(scale[2], 4))
        props_table.insert(loc=2, column="scale", value=f"{round_scale}")
    else: 
        props_table.insert(loc=2, column="scale", value=f"{tuple(np.ones(labels.ndim))}") 


    ######################################################
    ## optional: DISTRIBUTION OF CONTACTS MEASUREMENTS
    ######################################################
    if include_dist is True:
        XY_contact_dist, XY_bins, XY_wedges = get_XY_distribution(mask=mask, 
                                                                  obj=a_int_b,
                                                                  obj_name=f"{a_name}X{b_name}",
                                                                  centering_obj=dist_centering_obj,
                                                                  scale=scale,
                                                                  center_on=dist_center_on,
                                                                  keep_center_as_bin=dist_keep_center_as_bin,
                                                                  num_bins=dist_num_bins,
                                                                  zernike_degrees=dist_zernike_degrees)
        
        Z_contact_dist = get_Z_distribution(mask=mask,
                                            obj=a_int_b,
                                            obj_name=f"{a_name}X{b_name}",
                                            center_obj=dist_centering_obj,
                                            scale=scale)
        
        contact_dist_tab = pd.merge(XY_contact_dist, Z_contact_dist, on=["object", "scale"])

        return props_table, contact_dist_tab
    else:
        return props_table

# def get_aXb_stats_3D(a, b, mask, use_shell_a=False):
#     """
#     collect volumentric stats of `a` intersect `b`
#     """
#     properties = ["label"]  # our index to organelles
#     # add area
#     properties = properties + ["area", "equivalent_diameter"]
#     #  position:
#     properties = properties + ["centroid", "bbox"]  # ,  'weighted_centroid']
#     # etc
#     properties = properties + ["slice"]

#     # in case we sent a boolean mask (e.g. cyto, nucleus, cellmask)
#     a = _assert_uint16_labels(a)
#     b = _assert_uint16_labels(b)

#     if use_shell_a:
#         a_int_b = np.logical_and(np.logical_xor(a > 0, binary_erosion(a > 0)), b > 0)
#     else:
#         a_int_b = np.logical_and(a > 0, b > 0)

#     labels = label(apply_mask(a_int_b, mask)).astype("int")

#     props = regionprops_table(labels, intensity_image=None, properties=properties, extra_properties=None)

#     props["surface_area"] = surface_area_from_props(labels, props)
#     # there could be a bug here if there are spurious labels in the corners of the slices

#     label_a = []
#     index_ab = []
#     label_b = []
#     for index, lab in enumerate(props["label"]):
#         # this seems less elegant than you might wish, given that regionprops returns a slice,
#         # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
#         volume = labels[props["slice"][index]]
#         la = a[props["slice"][index]]
#         lb = b[props["slice"][index]]
#         volume = volume == lab
#         la = la[volume]
#         lb = lb[volume]

#         all_as = np.unique(la[la>0]).tolist()
#         all_bs = np.unique(lb[lb>0]).tolist()
#         if len(all_as) != 1:
#             print(f"we have an error.  as-> {all_as}")
#         if len(all_bs) != 1:
#             print(f"we have an error.  bs-> {all_bs}")

#         label_a.append(all_as[0] )
#         label_b.append(all_bs[0] )
#         index_ab.append(f"{all_as[0]}_{all_bs[0]}")


#     props["label_a"] = label_a #[np.unique(a[s])[:1].tolist() for s in props["slice"]]
#     props["label_b"] = label_b #[np.unique(b[s])[:1].tolist() for s in props["slice"]]
#     props_table = pd.DataFrame(props)
#     props_table.rename(columns={"area": "volume"}, inplace=True)
#     props_table.drop(columns="slice", inplace=True)
#     props_table.insert(loc=0,column='label_',value=index_ab)
#     props_table.insert(loc=0,column='shell',value=use_shell_a)

#     return props_table

### USED ###
def create_masked_sum_projection(img_in:np.ndarray, mask:Union[np.ndarray, None]=None, to_bool:bool=True) -> np.ndarray:
    """
    Parameters:
    ----------
    img_in:
        3D (ZYX) np.ndarray that will be summed along the Z axis
    mask:
        Optional - mask of the region you want to include in the final sum projection
    to_bool:
        True = input image is created in a boolean image before sum projection (useful for segmentation images where each object is coded as a unique ID number; like after skimage.segmentation.label())
        False = original input image is used for the sum projection
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

### USED ###
def get_XY_distribution(        
        mask: np.ndarray,
        centering_obj: np.ndarray,
        obj:np.ndarray,
        obj_name: str,
        scale: Union[tuple, None]=None,
        num_bins: Union[int, None] = None,
        center_on: bool = False,
        keep_center_as_bin: bool = True,
        zernike_degrees: Union[int,None] = None):

    """
    Params
    ----------
    mask_obj: np.ndarray,
        a binary 3D (ZYX) np.ndarray of the area that will be measured from
    centering_obj: np.ndarray
        a binary 3D (ZYX) np.ndarray of the object that will be used as the center of the concentric rins ("bins")
    obj: np.ndarray
        a 3D (ZYX) np.ndarray image of what will be measured within the masked area
    obj_name: str
        the name or nickname for the obj being measured; this will appear as a column in the output datasheet
    scale: Union[tuple, None]=None
        a tuple that contains the real world dimensions for each dimension in the image (Z, Y, X)
    num_bins: Union[int,None] = None
        the number of concentric rings to draw between the centering object and edge of the mask; None will result in 5 bins
    center_on: bool = False
        True = distribute the bins from the center of the centering object
        False = distribute the bins from the edge of the centering object
    keep_center_as_bin: bool = True
        True = include the centering object area when creating the bins
        False = do not include the centering object area when creating the bins
    zernike_degrees: Union[int,None] = None
        the number of zernike degrees to include for the zernike shape descriptors; if None, the zernike measurements will not 
        be included in the output


    Returns
    -----------
    XY_metrics:
        a pandas Dataframe of bin, wedge, and zernike measurements
    dist_bin_mask:
        an np.ndarray mask of the concentric ring bins
    dist_wedge_mask 
        an np.ndarray mask of the 8 radial wedges

    """

    mask_proj = create_masked_sum_projection(mask)
    center_proj = create_masked_sum_projection(centering_obj,mask.astype(bool))
    obj_proj = create_masked_sum_projection(obj,mask.astype(bool))
 

    XY_metrics, dist_bin_mask, dist_wedge_mask = get_concentric_distribution(mask_proj=mask_proj, 
                                                        centering_proj=center_proj, 
                                                        obj_proj=obj_proj, 
                                                        obj_name=obj_name, 
                                                        scale=scale,
                                                        bin_count=num_bins, 
                                                        center_on=center_on,
                                                        keep_center_as_bin=keep_center_as_bin)
    
    if zernike_degrees is not None:
        zernike_metrics = get_zernike_metrics(cellmask_proj=mask_proj, 
                                            org_proj=obj_proj,
                                            organelle_name=obj_name, 
                                            nucleus_proj=center_proj, 
                                            zernike_degree=zernike_degrees)
        
        XY_metrics = pd.merge(XY_metrics, zernike_metrics, on="object")

    return XY_metrics, dist_bin_mask, dist_wedge_mask 

# def get_radial_stats(        
#         cellmask_obj: np.ndarray,
#         organelle_mask: np.ndarray,
#         organelle_obj:np.ndarray,
#         organelle_img: np.ndarray,
#         organelle_name: str,
#         nuclei_obj: np.ndarray,
#         n_rad_bins: Union[int,None] = None,
#         n_zernike: Union[int,None] = None,
#         ):

#     """
#     Params


#     Returns
#     -----------
#     rstats table of radial distributions
#     zstats table of zernike magnitudes and phases
#     rad_bins image of the rstats bins over the cellmask_obj 

#     """


#     # flattened
#     cellmask_proj = create_masked_sum_projection(cellmask_obj) #test_cell_proj_a
#     org_proj = create_masked_sum_projection(organelle_obj,organelle_mask.astype(bool)) #test_colgi_proj_a
#     img_proj = create_masked_sum_projection(organelle_img,organelle_mask.astype(bool), to_bool=False)

#     nucleus_proj = create_masked_sum_projection(nuclei_obj,cellmask_obj.astype(bool)) #test_nuc_proj_a

#     radial_stats, radial_bin_mask = get_XY_distribution(cellmask_proj=cellmask_proj,
#                                                         nucleus_proj=nucleus_proj,
#                                                         org_proj=org_proj,
#                                                         org_name=organelle_name,
#                                                         bin_count=n_rad_bins,
#                                                         center_obj_as_bin = True,
#                                                         bins_from_center = False)
    
#     zernike_stats = get_zernike_stats(
#                                       cellmask_proj=cellmask_proj, 
#                                       org_proj=org_proj, 
#                                       img_proj=img_proj, 
#                                       organelle_name=organelle_name, 
#                                       nucleus_proj=nucleus_proj, 
#                                       zernike_degree = n_zernike
#                                       )

#     return radial_stats,zernike_stats,radial_bin_mask

### USED ###
def get_normalized_distance_and_mask(labels: np.ndarray, 
                                      center_objects: Union[np.ndarray, None], 
                                      center_on: bool):
    """
    helper for radial distribution
    Parameters:
    ----------
    labels:
        2D (YX) np.ndarray - normally the result of a binary ZYX segmentation of the cell mask after a sum projection across the Z dimension
    center_object:
        2D (YX) np.ndarray - normally the result of a binary ZYX segmentation of the nucleus after a sum projection across the Z dimension.
        If no centering object is included, the center of the labels will be used.
    center_on:
        True = the center of the centering object will be used as the starting point to calculate the distance from the center
        False = the edge of the centering object will be used as the starting point to calculate the distance from the center
    
    Output:
    ----------
    normalized_distance:
        2D (YX) np.ndarray with intensity values representing the distance btween the edge of the "labels" and the centering object
    good_mask:
        mask of the areas that were included in the normalized_distance output
    i_center
    j_center
    """

    d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)

    if center_objects is not None:
        center_labels = label(center_objects)
        pixel_counts = centrosome.cpmorphology.fixup_scipy_ndimage_result(ndi_sum(np.ones(center_labels.shape), 
                                                                                  center_labels, 
                                                                                  np.arange(1, np.max(center_labels) + 1, dtype=np.int32)))
        good = pixel_counts > 0
        i, j = (centrosome.cpmorphology.centers_of_labels(center_labels) + 0.5).astype(int)
        ig = i[good]
        jg = j[good]
        lg = np.arange(1, len(i) + 1)[good]
        
        if center_on:  # Reduce the propagation labels to the centers of the centering objects
            center_labels = np.zeros(center_labels.shape, int)
            center_labels[ig, jg] = lg

        cl, d_from_center = centrosome.propagate.propagate(np.zeros(center_labels.shape), center_labels, labels != 0, 1)
        cl[labels == 0] = 0

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

            iii, jjj = np.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
            di = iii[missing_mask] - i[cl[missing_mask] - 1]
            dj = jjj[missing_mask] - j[cl[missing_mask] - 1]
            d_from_center[missing_mask] = np.sqrt(di * di + dj * dj)

        good_mask = cl > 0
            
    else:
        i, j = centrosome.cpmorphology.maximum_position_of_labels(d_to_edge, labels, [1])
        center_labels = np.zeros(labels.shape, int)
        center_labels[i, j] = labels[i, j]
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

    i_center = np.zeros(cl.shape)
    i_center[good_mask] = i[cl[good_mask] - 1]

    j_center = np.zeros(cl.shape)
    j_center[good_mask] = j[cl[good_mask] - 1]

    normalized_distance = np.zeros(labels.shape)
    total_distance = d_from_center + d_to_edge
    normalized_distance[good_mask] = d_from_center[good_mask] / (total_distance[good_mask] + 0.001)
    
    return normalized_distance, good_mask, i_center, j_center

# def get_normalized_distance_and_mask(labels, center_objects, center_on_nuc, keep_nuc_bins):
#     """
#     helper for radial distribution
#     """
#     d_to_edge = centrosome.cpmorphology.distance_to_edge(labels) # made a local version
#     ## use the nucleus as the center 
#     if center_objects is not None:
#         # don't need to do this size_similarity trick.  I KNOW that labels and center_objects are the same size
#         # center_labels, cmask = size_similarly(labels, center_objects)
#         # going to leave them as labels, so in princi    ple the same code could work for partitioned masks (labels)
#         center_labels = label(center_objects)
#         pixel_counts = centrosome.cpmorphology.fixup_scipy_ndimage_result(
#                             ndi_sum(
#                                 np.ones(center_labels.shape),
#                                 center_labels,
#                                 np.arange(
#                                     1, np.max(center_labels) + 1, dtype=np.int32
#                                 ),
#                             )
#                         )
#         good = pixel_counts > 0
#         i, j = ( centrosome.cpmorphology.centers_of_labels(center_labels) + 0.5).astype(int)
#         ig = i[good]
#         jg = j[good]
#         lg = np.arange(1, len(i) + 1)[good]
        
#         if center_on_nuc:  # Reduce the propagation labels to the centers of the centering objects
#             center_labels = np.zeros(center_labels.shape, int)
#             center_labels[ig, jg] = lg


#         cl, d_from_center = centrosome.propagate.propagate(  np.zeros(center_labels.shape), center_labels, labels != 0, 1)
#         cl[labels == 0] = 0            # Erase the centers that fall outside of labels


#         # SHOULD NEVER NEED THIS because we arent' looking at multiple
#         # If objects are hollow or crescent-shaped, there may be objects without center labels. As a backup, find the
#         # center that is the closest to the center of mass.
#         missing_mask = (labels != 0) & (cl == 0)
#         missing_labels = np.unique(labels[missing_mask])
        
#         if len(missing_labels):
#             print("WTF!!  how did we have missing labels?")
#             all_centers = centrosome.cpmorphology.centers_of_labels(labels)
#             missing_i_centers, missing_j_centers = all_centers[:, missing_labels-1]
#             di = missing_i_centers[:, np.newaxis] - ig[np.newaxis, :]
#             dj = missing_j_centers[:, np.newaxis] - jg[np.newaxis, :]
#             missing_best = lg[np.argsort(di * di + dj * dj)[:, 0]]
#             best = np.zeros(np.max(labels) + 1, int)
#             best[missing_labels] = missing_best
#             cl[missing_mask] = best[labels[missing_mask]]

#             # Now compute the crow-flies distance to the centers of these pixels from whatever center was assigned to the object.
#             iii, jjj = np.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
#             di = iii[missing_mask] - i[cl[missing_mask] - 1]
#             dj = jjj[missing_mask] - j[cl[missing_mask] - 1]
#             d_from_center[missing_mask] = np.sqrt(di * di + dj * dj)

#         good_mask = cl > 0

#         if not keep_nuc_bins:
#             # include pixels within the centering objects
#             # when performing calculations from the centers
#             good_mask = good_mask & (center_labels == 0)
            
#     # else: # ELSE     if center_objects is  None so center on the middle of the cellmask_mask
#     #     i, j = centrosome.cpmorphology.maximum_position_of_labels(   d_to_edge, labels, [1])
#     #     center_labels = np.zeros(labels.shape, int)
#     #     center_labels[i, j] = labels[i, j]
#     #     # Use the coloring trick here to process touching objectsin separate operations
#     #     colors = centrosome.cpmorphology.color_labels(labels)
#     #     ncolors = np.max(colors)
#     #     d_from_center = np.zeros(labels.shape)
#     #     cl = np.zeros(labels.shape, int)

#     #     for color in range(1, ncolors + 1):
#     #         mask = colors == color
#     #         l, d = centrosome.propagate.propagate( np.zeros(center_labels.shape), center_labels, mask, 1)
#     #         d_from_center[mask] = d[mask]
#     #         cl[mask] = l[mask]

#     #     good_mask = cl > 0

#     ## define spatial distribution from masks
#     # collect arrays of centers
#     i_center = np.zeros(cl.shape)
#     i_center[good_mask] = i[cl[good_mask] - 1]
#     j_center = np.zeros(cl.shape)
#     j_center[good_mask] = j[cl[good_mask] - 1]

#     normalized_distance = np.zeros(labels.shape)
#     total_distance = d_from_center + d_to_edge

#     normalized_distance[good_mask] = d_from_center[good_mask] / ( total_distance[good_mask] + 0.001 )
#     return normalized_distance, good_mask, i_center, j_center

# def get_XY_distribution(
#         cellmask_proj: np.ndarray,
#         nucleus_proj: np.ndarray,
#         org_proj: np.ndarray,
#         org_name: str,
#         bin_count: Union[int, None] = 5,
#         center_obj_as_bin: bool = True,
#         bins_from_center:bool = False
#     ):
#     """
#     Based on CellProfiler's measureobjectintensitydistribution. Measure the distribution of segmented objects within a masked area. 
#     In our case, we will usually utilize this function to measure the amount of an organelle within the cell.
#     Radial bins are created out from a center point, usually the nucleus edge.

#     Parameters
#     ------------
#     cellmask_proj: np.ndarray
#         a sum projection of the segmented cell area where the "intensity" value of each pixel is equal to the number of z slices where the binary cell mask is True
#     nucleus_proj: np.ndarray
#         a sum projection of the segmented nucleus area where the "intensity" value of each pixel is equal to the number of z slices where the binary nucleus mask is True
#     org_proj: np.ndarray,
#         a sum projection of the segmented organelle area where the "intensity" value of each pixel is equal to the number of z slices where the binary organelle mask is True
#     org_name: str,
#         the name or nickname of your organelle; used for labeling columns in the dataframe
#     bin_count: Union[int, None] = 5,
#         the number of bins to create within the cell mask
#     center_obj_as_bin: bool = True,
#         True = include the centering object area when creating the bins
#         False = do not include the centering object area when creating the bins
#     bins_from_center:bool = False
#         True = distribute the bins from the center of the centering object
#     masked


#     Returns
#     -------------
#     returns one statistics table (pd.DataFrame) + bin_array (np.ndarray) image
#     """

#     # other parameters that will stay constant
#     nobjects = 1

#     # create binary arrays
#     center_objects = nucleus_proj>0 
#     cellmask = (cellmask_proj>0).astype(np.uint16)


#     ################   ################
#     ## define masks for computing distances
#     ################   ################
#     normalized_distance, good_mask, i_center, j_center = get_normalized_distance_and_mask(cellmask, center_objects, bins_from_center, center_obj_as_bin)
    
#     if normalized_distance is None:
#         print('WTF!!  normalized_distance returned wrong')

#     ################   ################
#     ## get histograms
#     ################   ################
#     ngood_pixels = np.sum(good_mask)
#     good_labels = cellmask[good_mask]

#     # protect against None normaized_distances
#     bin_array = (normalized_distance * bin_count).astype(int)
#     bin_array[bin_array > bin_count] = bin_count # shouldn't do anything

#     #                 (    i          ,         j              )
#     labels_and_bins = (good_labels - 1, bin_array[good_mask])

#     #                coo_matrix( (             data,             (i, j)    ), shape=                      )
#     histogram_cmsk = coo_matrix( (cellmask_proj[good_mask], labels_and_bins), shape=(nobjects, bin_count) ).toarray()
#     histogram_org = coo_matrix(  (org_proj[good_mask],      labels_and_bins), shape=(nobjects, bin_count) ).toarray()

#     bin_array = (normalized_distance * bin_count).astype(int)

#     sum_by_object_cmsk = np.sum(histogram_cmsk, 1) # flattened cellmask voxel count
#     sum_by_object_org = np.sum(histogram_org, 1)  # organelle voxel count


#     # DEPRICATE: since we are NOT computing object_i by object_i (individual organelle labels)
#     # sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]
#     # fraction_at_distance = histogram / sum_by_object_per_bin

#     # number of bins.
#     number_at_distance = coo_matrix(( np.ones(ngood_pixels), labels_and_bins), (nobjects, bin_count)).toarray()

#     # sicne we aren't breaking objects apart this is just ngood_pixels

#     sum_by_object = np.sum(number_at_distance, 1)

#     sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count))[0]
#     fraction_at_bin = number_at_distance / sum_by_object_per_bin # sums to 1.0

#     # object_mask = number_at_distance > 0
#     # DEPRICATE:# not doing over multiple objects so don't need object mask.. or fractionals
#     # mean_pixel_fraction = fraction_at_distance / ( fraction_at_bin + np.finfo(float).eps )
#     # masked_fraction_at_distance = np.ma.masked_array( fraction_at_distance, ~object_mask )
#     # masked_mean_pixel_fraction = np.ma.masked_array(mean_pixel_fraction, ~object_mask)

#     ################   ################
#     ## collect Anisotropy calculation.  + summarize
#     ################   ################
#     # Split each cell into eight wedges, then compute coefficient of variation of the wedges' mean intensities
#     # in each ring. Compute each pixel's delta from the center object's centroid
#     i, j = np.mgrid[0 : cellmask.shape[0], 0 : cellmask.shape[1]]
#     imask = i[good_mask] > i_center[good_mask]
#     jmask = j[good_mask] > j_center[good_mask]
#     absmask = abs(i[good_mask] - i_center[good_mask]) > abs(j[good_mask] - j_center[good_mask])
#     radial_index = (imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4)

#     # return radial_index, labels, good_mask, bin_indexes
#     stat_names =[]
#     cv_cmsk = []
#     cv_obj = []

#     # collect the numbers from each "bin"
#     for bin in range(bin_count):
#         bin_mask = good_mask & (bin_array == bin)
#         bin_pixels = np.sum(bin_mask)

#         bin_labels = cellmask[bin_mask]

#         bin_radial_index = radial_index[bin_array[good_mask] == bin]
#         labels_and_radii = (bin_labels - 1, bin_radial_index)
#         pixel_count = coo_matrix( (np.ones(bin_pixels), labels_and_radii), (nobjects, 8) ).toarray()

#         radial_counts_cmsk = coo_matrix( (cellmask_proj[bin_mask], labels_and_radii), (nobjects, 8) ).toarray()
#         radial_counts = coo_matrix( (org_proj[bin_mask], labels_and_radii), (nobjects, 8) ).toarray()
#         # radial_values = coo_matrix( (img_proj[bin_mask], labels_and_radii), (nobjects, 8) ).toarray()

#         # we might need the masked arrays for some organelles... but I think not. keeping for now
#         mask = pixel_count == 0

#         radial_means_cmsk = np.ma.masked_array(radial_counts_cmsk / pixel_count, mask)
#         radial_cv_cmsk = np.std(radial_means_cmsk, 1) / np.mean(radial_means_cmsk, 1)
#         radial_cv_cmsk[np.sum(~mask, 1) == 0] = 0
#         radial_cv_cmsk.mask = np.sum(~mask, 1) == 0


#         radial_means_obj = np.ma.masked_array(radial_counts / pixel_count, mask)
#         radial_cv_obj = np.std(radial_means_obj, 1) / np.mean(radial_means_obj, 1)
#         radial_cv_obj[np.sum(~mask, 1) == 0] = 0
#         radial_cv_obj.mask = np.sum(~mask, 1) == 0

#         bin_name = str(bin + 1) if bin > 0 else "1"

#         stat_names.append(bin_name)
#         cv_cmsk.append(float(np.mean(radial_cv_cmsk)))  #convert to float to make importing from csv more straightforward
#         cv_obj.append(float(np.mean(radial_cv_obj)))
    
#     stats_dict={'organelle': org_name,
#                 'mask': 'cell',
#                 'radial_n_bins': bin_count,
#                 'radial_bins': [stat_names],
#                 'radial_cm_vox_cnt': [histogram_cmsk.squeeze().tolist()],
#                 'radial_org_vox_cnt': [histogram_org.squeeze().tolist()],
#                 # 'radial_org_intensity': [histogram_img.squeeze().tolist()],
#                 'radial_n_pix': [number_at_distance.squeeze().tolist()],
#                 'radial_cm_cv':[cv_cmsk],
#                 'radial_org_cv':[cv_obj]}

#     # stats_tab = pd.DataFrame(statistics,columns=col_names)
#     stats_tab = pd.DataFrame(stats_dict)  
#     return stats_tab, bin_array

### USED ###
def get_concentric_distribution(
        mask_proj: np.ndarray,
        centering_proj: np.ndarray,
        obj_proj: np.ndarray,
        obj_name: str,
        bin_count: int,
        center_on: bool = False,
        keep_center_as_bin: bool = True,
        scale: Union[tuple, None]=None):
    """
    Based on CellProfiler's measureobjectintensitydistribution. Measure the distribution of segmented objects within a masked area. 
    In our case, we will usually utilize this function to measure the amount of an organelle within the cell.
    Radial bins are created out from a center point, usually the nucleus edge.

    
    Parameters
    ------------
    mask_proj: np.ndarray
        a sum projection of the region you want to measure the distribution from where the "intensity" value of each pixel is equal 
        to the number of z slices where the binary cell mask is True
    centering_proj: np.ndarray
        a sum projection of the object you want to use as the center of the distribution where the "intensity" value of each pixel is 
        equal to the number of z slices where the binary nucleus mask is True
    obj_proj: np.ndarray,
        a sum projection of the stuff you want to measure where the "intensity" value of each pixel is equal to the number of z slices 
        where the binary organelle mask is True (for a segmented image) or the total intensity at that point (for a gray scale image)
    obj_name: str,
        the name or nickname of your object being measured; used for labeling columns in the dataframe
    bin_count: int,
        the number of concentric rings, or "bins", to create within the mask
    center_on: bool = False,
        True = distribute the bins from the center of the centering object
        False = distribute the bins from the edge of the centering object
    keep_center_as_bin: bool = True
        True = include the centering object area when creating the bins
        False = do not include the centering object area when creating the bins
    scale: Union[tuple, None]=None
        a tuple of floats representing the real-world dimensions for each image dimension (ZYX)
        

    Measurements
    ------------
    If scale is used, "vox_cnt" is replaced by "vol" and "n_pix_ is replaced by "area" in the title below.

    object: the nickname of what is being measured (e.g., golgi, golgiXER, ER_img)
    XY_n_bins: number of bins
    XY_bins: list of bin number
    XY_mask_vox_cnt_perbin: number of voxels in the 3D cell mask per bin
    XY_obj_vox_cnt_perbin: number of voxels of the 3D object per bin
    XY_center_vox_cnt_perbin: number of voxels of the 3D centering object per bin
    XY_n_pix_perbin: number of pixels per bin in the XY mask
    XY_portion_pix_perbin: the portion of pixels in the XY mask per bin
    XY_n_wedges: number of wedges
    XY_wedges: list of wedge numbers
    XY_mask_vox_cnt_perwedge: number of voxels in the 3D cell mask per wedge
    XY_obj_vox_cnt_perwedge: number of voxels of the 3D object per wedge
    XY_center_vox_cnt_perwedge: number of voxels of the 3D centering object per wedge
    XY_n_pix_perwedge: number of pixels per wedge in the XY mask
    XY_portion_pix_perwedge: the portion of pixels in the XY mask per bin
    XY_wedges_perbin: list of wedges that have >0 pixels in the mask for all bins
    XY_mask_vox_cnt_wedges_perbin: number of voxels in the 3D cell mask per wedge per bin
    XY_obj_vox_cnt_wedges_perbin:number of voxels of the 3D object per wedge per bin
    XY_center_vox_cnt_wedges_perbin: number of voxels of the 3D centering object per wedge per bin
    XY_n_pix_wedges_perbin: number of pixels per wedge per bin in the XY mask
    XY_mask_cv_perbin: the coefficient of variance of the wedges within each bin for the mask
    XY_obj_cv_perbin: the coefficient of variance of the wedges within each bin for the object segmentation
    XY_center_cv_perbin: the coefficient of variance of the wedges within each bin for the centering object

    
    Returns
    -------------
    tab: (pd.DataFrame) table of measurements of the object distribution
    bin_array: (np.ndarray) mask of the concentric rings to measure distribution from
    wedge_array: (np.ndarray) mask of the wedges (pie slices) that divide each bin into 8 parts
    """
    # other parameters that will stay constant
    nobjects = 1

    # create binary arrays
    center_objects = centering_proj>0 
    mask = (mask_proj>0).astype(np.uint16)


    ################   ################
    ## compute distances and make bins and wedges masks
    ################   ################
    # created normalized distances
    normalized_distance, good_mask, i_center, j_center = get_normalized_distance_and_mask(labels=mask, center_objects=center_objects, center_on=center_on)
    if normalized_distance is None:
        print('WTF!!  normalized_distance returned wrong')

    # create bin mask array
    if keep_center_as_bin:
        if center_on:
            bin_array = (normalized_distance * bin_count).astype(int)
        else:
            bin_array= ((normalized_distance * (bin_count-1))+1).astype(int)
            bin_array[center_objects]=0
            bin_array[~good_mask]=0
    else:
        good_mask[center_objects]=0
        if center_on:
            normalized_distance[good_mask] = (normalized_distance[good_mask] - normalized_distance[good_mask].min())/(normalized_distance[good_mask].max() - normalized_distance[good_mask].min())
        bin_array = (normalized_distance * bin_count).astype(int)
            
    bin_array[bin_array > bin_count] = bin_count
    
    # create wedges mask array
    i, j = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    imask = i[good_mask] > i_center[good_mask]
    jmask = j[good_mask] > j_center[good_mask]
    absmask = abs(i[good_mask] - i_center[good_mask]) > abs(j[good_mask] - j_center[good_mask])
    radial_index = (imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4)

    wedge_array = np.zeros_like(good_mask, dtype=int)
    wedge_array[good_mask] = radial_index
    

    ################   ################
    ## get histograms
    ################   ################
    ngood_pixels = np.sum(good_mask)
    good_labels = mask[good_mask]

    # whole cell bin and wedge measurements
    mask_arrays = [bin_array, wedge_array]
    sections = [bin_count, 8]
    types = ['bin', 'wedge']

    met_dict = {}

    for array, num, name in zip(mask_arrays, sections, types):
        labels_and_bins = (good_labels - 1, array[good_mask])

        met_dict[f"XY_mask_vox_cnt_per{name}"] = [coo_matrix((mask_proj[good_mask], labels_and_bins), shape=(nobjects, num)).toarray().squeeze().tolist()]
        met_dict[f"XY_obj_vox_cnt_per{name}"] = [coo_matrix((obj_proj[good_mask], labels_and_bins), shape=(nobjects, num)).toarray().squeeze().tolist()]
        met_dict[f"XY_center_vox_cnt_per{name}"] = [coo_matrix((centering_proj[good_mask], labels_and_bins), shape=(nobjects, num)).toarray().squeeze().tolist()]
        n_pixels = [coo_matrix((np.ones(ngood_pixels), labels_and_bins), (nobjects, num)).toarray().squeeze().tolist()]
        met_dict[f"XY_n_pix_per{name}"] = n_pixels

        total_pixels = np.sum(n_pixels, 1)
        total_repeated = np.dstack([total_pixels] * (num))[0]
        met_dict[f"XY_portion_pix_per{name}"] = [(n_pixels / total_repeated).squeeze().tolist()]


    # per wedge per bin measurements
    bin_names =[]
    cv_mask = []
    cv_obj = []
    cv_center = []
    mask_wedge_perbin = []
    obj_wedge_perbin = []
    center_wedge_perbin = []
    pxl_cnt_wedge_perbin = []
    wedges_perbin = []

    for bin in range(bin_count):
        bin_mask = good_mask & (bin_array == bin)
        bin_pixels = np.sum(bin_mask)

        bin_labels = mask[bin_mask]

        bin_radial_index = radial_index[bin_array[good_mask] == bin]
        labels_and_radii = (bin_labels - 1, bin_radial_index)

        radial_counts_mask = coo_matrix((mask_proj[bin_mask], labels_and_radii), (nobjects, 8) ).toarray()
        radial_counts_obj = coo_matrix((obj_proj[bin_mask], labels_and_radii), (nobjects, 8)).toarray()
        radial_counts_center = coo_matrix((centering_proj[bin_mask], labels_and_radii), (nobjects, 8)).toarray()
        pixel_count = coo_matrix((np.ones(bin_pixels), labels_and_radii), (nobjects, 8)).toarray()

        n_mask = pixel_count == 0

        radial_counts = [radial_counts_mask, radial_counts_obj, radial_counts_center]
        radial_cvs = []
        for count in radial_counts:
            radial_norm = np.ma.masked_array(count / pixel_count, n_mask)
            radial_cv = np.std(radial_norm, 1) / np.mean(radial_norm, 1)
            radial_cv[np.sum(~n_mask, 1) == 0] = 0
            radial_cv.mask = np.sum(~n_mask, 1) == 0
            radial_cvs.append(radial_cv)

        bin_name = bin + 1 if bin > 0 else 1
        wedges_perbin_name = np.ma.masked_array([it+1 for it in range(8)])

        bin_names.append(bin_name)
        cv_mask.append(float(np.mean(radial_cvs[0])))
        cv_obj.append(float(np.mean(radial_cvs[1])))
        cv_center.append(float(np.mean(radial_cvs[2])))
        mask_wedge_perbin.append(radial_counts[0].squeeze().tolist())
        obj_wedge_perbin.append(radial_counts[1].squeeze().tolist())
        center_wedge_perbin.append(radial_counts[2].squeeze().tolist())
        pxl_cnt_wedge_perbin.append(pixel_count.squeeze().tolist())
        wedges_perbin.append(wedges_perbin_name.data.squeeze().tolist())
    

    ################   ################
    ## create data table and account for scale
    ################   ################
    met_dict_1 = {'object': obj_name,
                  'XY_n_bins': bin_count,
                  'XY_bins': [bin_names]}
    met_dict_2 = dict(list(met_dict.items())[:5])
    met_dict_3 = {'XY_n_wedges': 8,
                  'XY_wedges': str([it+1 for it in range(8)])}
    met_dict_4 = dict(list(met_dict.items())[5:])
    met_dict_5 = {'XY_wedges_perbin': [wedges_perbin],
                  'XY_mask_vox_cnt_wedges_perbin':[mask_wedge_perbin],
                  'XY_obj_vox_cnt_wedges_perbin':[obj_wedge_perbin],
                  'XY_center_vox_cnt_wedges_perbin': [center_wedge_perbin],
                  'XY_n_pix_wedges_perbin': [pxl_cnt_wedge_perbin],
                  'XY_mask_cv_perbin':[cv_mask],
                  'XY_obj_cv_perbin':[cv_obj],
                  'XY_center_cv_perbin': [cv_center]}
    
    dict_combined = dict(itertools.chain(met_dict_1.items(), met_dict_2.items(), met_dict_3.items(), met_dict_4.items(), met_dict_5.items()))
    tab = pd.DataFrame(dict_combined)

    # account for scale
    if scale is not None:
        round_scale = (round(scale[0], 4), round(scale[1], 4), round(scale[2], 4))
        tab.insert(loc=1, column="scale", value=f"{round_scale}")
        
        # measurements affected by scale
        vol_mets = ['XY_mask_vox_cnt_perbin', 'XY_obj_vox_cnt_perbin', 'XY_center_vox_cnt_perbin', 'XY_mask_vox_cnt_perwedge', 'XY_obj_vox_cnt_perwedge',
                    'XY_center_vox_cnt_perwedge', 'XY_mask_vox_cnt_wedges_perbin', 'XY_obj_vox_cnt_wedges_perbin', 'XY_center_vox_cnt_wedges_perbin']
        area_mets = ['XY_n_pix_perbin', 'XY_n_pix_perwedge', 'XY_n_pix_wedges_perbin']

        for met in vol_mets:
            tab[met.replace('_vox_cnt_', "_vol_")] = [(np.float_(tab[met][0]) * np.prod(scale)).squeeze().tolist()]
        for met in area_mets:
            tab[met.replace('_n_pix_', "_area_")] = [(np.float_(tab[met][0]) * np.prod(scale[1:])).squeeze().tolist()]
    else: 
        tab.insert(loc=2, column="scale", value=f"{tuple(np.ones(3))}")

    return tab, bin_array, wedge_array

# def get_XY_distribution(
#         cellmask_proj: np.ndarray,
#         org_proj: np.ndarray,
#         img_proj: np.ndarray,
#         org_name: str,
#         nucleus_proj: np.ndarray,
#         n_bins: int = 5,
#         from_edges: bool = True,
#     ):
#     """Perform the radial measurements on the image set

#     Parameters
#     ------------
#     cellmask_proj: np.ndarray,
#     org_proj: np.ndarray,
#     img_proj: np.ndarray,
#     org_name: str,
#     nucleus_proj: Union[np.ndarray, None],
#     n_bins: int = 5,
#     from_edges: bool = True,

#     masked

#     # params
#     #   n_bins .e.g. 6
#     #   normalizer - cellmask_voxels, organelle_voxels, cellmask_and_organelle_voxels
#     #   from_edges = True


#     Returns
#     -------------
#     returns one statistics table + bin_indexes image array
#     """

#     # other params
#     bin_count = n_bins if n_bins is not None else 5
#     nobjects = 1
#     scale_bins = True 
#     keep_nuc_bins = True # this toggles whether to count things inside the nuclei mask.  
#     center_on_nuc = False # choosing the edge of the nuclei or the center as the center to propogate from

#     center_objects = nucleus_proj>0 

#     # labels = label(cellmask_proj>0) #extent as 0,1 rather than bool    
#     labels = (cellmask_proj>0).astype(np.uint16)
#     # labels = np.zeros_like(cellmask_proj)
#     # labels[labels>0]=1

#     ################   ################
#     ## define masks for computing distances
#     ################   ################
#     normalized_distance, good_mask, i_center, j_center = get_normalized_distance_and_mask(labels, center_objects, center_on_nuc, keep_nuc_bins)
    
#     if normalized_distance is None:
#         print('WTF!!  normailzed_distance returned wrong')

#     ################   ################
#     ## get histograms
#     ################   ################
#     ngood_pixels = np.sum(good_mask)
#     good_labels = labels[good_mask]

#     # protect against None normaized_distances
#     bin_indexes = (normalized_distance * bin_count).astype(int)
#     bin_indexes[bin_indexes > bin_count] = bin_count # shouldn't do anything

#     #                 (    i          ,         j              )
#     labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

#     #                coo_matrix( (             data,             (i, j)    ), shape=                      )
#     histogram_cmsk = coo_matrix( (cellmask_proj[good_mask], labels_and_bins), shape=(nobjects, bin_count) ).toarray()
#     histogram_org = coo_matrix(  (org_proj[good_mask],      labels_and_bins), shape=(nobjects, bin_count) ).toarray()
#     histogram_img = coo_matrix(  (img_proj[good_mask],      labels_and_bins), shape=(nobjects, bin_count) ).toarray()

#     bin_indexes = (normalized_distance * bin_count).astype(int)

#     sum_by_object_cmsk = np.sum(histogram_cmsk, 1) # flattened cellmask voxel count
#     sum_by_object_org = np.sum(histogram_org, 1)  # organelle voxel count
#     sum_by_object_img = np.sum(histogram_img, 1)  # image intensity projection

#     # DEPRICATE: since we are NOT computing object_i by object_i (individual organelle labels)
#     # sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count + 1))[0]
#     # fraction_at_distance = histogram / sum_by_object_per_bin

#     # number of bins.
#     number_at_distance = coo_matrix(( np.ones(ngood_pixels), labels_and_bins), (nobjects, bin_count)).toarray()

#     # sicne we aren't breaking objects apart this is just ngood_pixels

#     sum_by_object = np.sum(number_at_distance, 1)

#     sum_by_object_per_bin = np.dstack([sum_by_object] * (bin_count))[0]
#     fraction_at_bin = number_at_distance / sum_by_object_per_bin # sums to 1.0

#     # object_mask = number_at_distance > 0
#     # DEPRICATE:# not doing over multiple objects so don't need object mask.. or fractionals
#     # mean_pixel_fraction = fraction_at_distance / ( fraction_at_bin + np.finfo(float).eps )
#     # masked_fraction_at_distance = np.ma.masked_array( fraction_at_distance, ~object_mask )
#     # masked_mean_pixel_fraction = np.ma.masked_array(mean_pixel_fraction, ~object_mask)

#     ################   ################
#     ## collect Anisotropy calculation.  + summarize
#     ################   ################
#     # Split each cell into eight wedges, then compute coefficient of variation of the wedges' mean intensities
#     # in each ring. Compute each pixel's delta from the center object's centroid
#     i, j = np.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
#     imask = i[good_mask] > i_center[good_mask]
#     jmask = j[good_mask] > j_center[good_mask]
#     absmask = abs(i[good_mask] - i_center[good_mask]) > abs(
#         j[good_mask] - j_center[good_mask]
#     )
#     radial_index = (
#         imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4
#     )

#     # return radial_index, labels, good_mask, bin_indexes
#     statistics = []
#     stat_names =[]
#     cv_cmsk = []
#     cv_obj = []
#     cv_img = []
#     # collect the numbers from each "bin"
#     for bin in range(bin_count):
#         bin_mask = good_mask & (bin_indexes == bin)
#         bin_pixels = np.sum(bin_mask)

#         bin_labels = labels[bin_mask]

#         bin_radial_index = radial_index[bin_indexes[good_mask] == bin]
#         labels_and_radii = (bin_labels - 1, bin_radial_index)
#         pixel_count = coo_matrix( (np.ones(bin_pixels), labels_and_radii), (nobjects, 8) ).toarray()

#         radial_counts_cmsk = coo_matrix( (cellmask_proj[bin_mask], labels_and_radii), (nobjects, 8) ).toarray()
#         radial_counts = coo_matrix( (org_proj[bin_mask], labels_and_radii), (nobjects, 8) ).toarray()
#         radial_values = coo_matrix( (img_proj[bin_mask], labels_and_radii), (nobjects, 8) ).toarray()

#         # we might need the masked arrays for some organelles... but I think not. keeping for now
#         mask = pixel_count == 0

#         radial_means_cmsk = np.ma.masked_array(radial_counts_cmsk / pixel_count, mask)
#         radial_cv_cmsk = np.std(radial_means_cmsk, 1) / np.mean(radial_means_cmsk, 1)
#         radial_cv_cmsk[np.sum(~mask, 1) == 0] = 0
#         radial_cv_cmsk.mask = np.sum(~mask, 1) == 0


#         radial_means_obj = np.ma.masked_array(radial_counts / pixel_count, mask)
#         radial_cv_obj = np.std(radial_means_obj, 1) / np.mean(radial_means_obj, 1)
#         radial_cv_obj[np.sum(~mask, 1) == 0] = 0
#         radial_cv_obj.mask = np.sum(~mask, 1) == 0

#         radial_means_img = np.ma.masked_array(radial_values / pixel_count, mask)
#         radial_cv_img = np.std(radial_means_img, 1) / np.mean(radial_means_img, 1)
#         radial_cv_img[np.sum(~mask, 1) == 0] = 0
#         radial_cv_img.mask = np.sum(~mask, 1) == 0

#         bin_name = str(bin) if bin > 0 else "Ctr"

#         # # there's gotta be a better way to collect this stuff together... pandas?
#         # statistics += [
#         #     (   bin_name,
#         #         # np.mean(number_at_distance[:, bin]), 
#         #         # np.mean(histogram_cmsk[:, bin]), 
#         #         # np.mean(histogram_org[:, bin]), 
#         #         # np.mean(histogram_img[:, bin]), 
#         #         np.mean(radial_cv_cmsk) ,
#         #         np.mean(radial_cv_obj) ,
#         #         np.mean(radial_cv_img) )
#         # ]
#         stat_names.append(bin_name)
#         cv_cmsk.append(float(np.mean(radial_cv_cmsk)))  #convert to float to make importing from csv more straightforward
#         cv_obj.append(float(np.mean(radial_cv_obj)))
#         cv_img.append(float(np.mean(radial_cv_obj)))

#     # TODO: fix this grooooos hack
#     # col_names=['organelle','mask','bin','n_bins','n_pix','cm_vox_cnt','org_vox_cnt','org_intensity','cm_radial_cv','org_radial_cv','img_radial_cv']
#     # stats_dict={'organelle': org_name,
#     #             'mask': 'cell',
#     #             'radial_n_bins': bin_count,
#     #             'radial_bins': [[s[0] for s in statistics]],
#     #             'radial_cm_vox_cnt': [histogram_cmsk.squeeze().tolist()],
#     #             'radial_org_vox_cnt': [histogram_org.squeeze().tolist()],
#     #             'radial_org_intensity': [histogram_img.squeeze().tolist()],
#     #             'radial_n_pix': [number_at_distance.squeeze().tolist()],
#     #             'radial_cm_cv':[[s[1] for s in statistics]],
#     #             'radial_org_cv':[[s[2] for s in statistics]],
#     #             'radial_img_cv':[[s[3] for s in statistics]],
#     #             }
    
#     stats_dict={'organelle': org_name,
#                 'mask': 'cell',
#                 'radial_n_bins': bin_count,
#                 'radial_bins': [stat_names],
#                 'radial_cm_vox_cnt': [histogram_cmsk.squeeze().tolist()],
#                 'radial_org_vox_cnt': [histogram_org.squeeze().tolist()],
#                 'radial_org_intensity': [histogram_img.squeeze().tolist()],
#                 'radial_n_pix': [number_at_distance.squeeze().tolist()],
#                 'radial_cm_cv':[cv_cmsk],
#                 'radial_org_cv':[cv_obj],
#                 'radial_img_cv':[cv_img],
#                 }

#     # stats_tab = pd.DataFrame(statistics,columns=col_names)
#     stats_tab = pd.DataFrame(stats_dict)  
#     return stats_tab, bin_indexes

### USED ###
def create_masked_depth_projection(img_in:np.ndarray, mask:Union[np.ndarray, None]=None, to_bool:bool=True) -> np.ndarray:
    """
    create a masked projection by summing together all XY pixels per Z plane/slice
    """
    img_out = img_in.astype(bool) if to_bool else img_in
    if mask is not None:
        img_out = apply_mask(img_out, mask)
    
    return img_out.sum(axis=(1,2))

### USED ###
def get_Z_distribution(        
        mask: np.ndarray,
        obj:np.ndarray,
        obj_name: str,
        center_obj: Union[np.ndarray, None],
        scale: Union[tuple, None] = None
        ):
    """
    quantification of distribution along the Z axis; all XY pixels are summed together per Z slice and then quantified

    Parameters
    ------------
    mask_obj: np.ndarray,
        a binary 3D (ZYX) np.ndarray of the area that will be measured from
    obj: np.ndarray
        a 3D (ZYX) np.ndarray image of what will be measured within the masked area
    obj_name: str
        the name or nickname for the obj being measured; this will appear as a column in the output datasheet
    centering_obj: np.ndarray
        optional - a binary 3D (ZYX) np.ndarray utilized as the center/reference point of the area; for cells, this is usually the nucleus
    scale: Union[tuple, None]=None
        a tuple that contains the real world dimensions for each dimension in the image (Z, Y, X)

    Returns
    -----------
    Z_tab:
        a pandas Dataframe of measurements for each z slice

    """

    # flattened
    mask_proj = create_masked_depth_projection(mask)
    obj_proj = create_masked_depth_projection(obj, mask.astype(bool))
    center_proj = create_masked_depth_projection(center_obj, mask.astype(bool)) if center_obj is not None else None

    Zdist_tab = pd.DataFrame({'object':obj_name,
                            'Z_n_slices':mask.shape[0],
                            'Z_slices':[[i for i in range(mask.shape[0])]],
                            'Z_mask_vox_cnt':[mask_proj.tolist()],
                            'Z_obj_vox_cnt':[obj_proj.tolist()],
                            'Z_center_vox_cnt':[center_proj.tolist()]})
    
    if scale is not None:
        round_scale = (round(scale[0], 4), round(scale[1], 4), round(scale[2], 4))
        Zdist_tab.insert(loc=1, column="scale", value=f"{round_scale}")

        Zdist_tab['Z_height'] = mask.shape[0] * scale[0]
        Zdist_tab['Z_mask_volume'] = [(mask_proj * np.prod(scale)).tolist()]
        Zdist_tab['Z_obj_volume'] = [(obj_proj * np.prod(scale)).tolist()]
        Zdist_tab['Z_center_volume'] = [(center_proj * np.prod(scale)).tolist()]
    else: 
        Zdist_tab.insert(loc=2, column="scale", value=f"{tuple(np.ones(3))}")

    return Zdist_tab

# def get_depth_stats(        
#         cellmask_obj: np.ndarray,
#         organelle_mask: np.ndarray,
#         organelle_obj:np.ndarray,
#         organelle_img: np.ndarray,
#         organelle_name: str,
#         nuclei_obj: Union[np.ndarray, None],
#         ):
#     """

#     """

#     # flattened
#     cellmask_proj = create_masked_depth_projection(cellmask_obj)
#     org_proj = create_masked_depth_projection(organelle_obj,organelle_mask.astype(bool))
#     img_proj = create_masked_depth_projection(organelle_img,organelle_mask.astype(bool), to_bool=False)

#     nucleus_proj = create_masked_depth_projection(nuclei_obj,cellmask_obj.astype(bool)) if nuclei_obj is not None else None
#     z_bins = [i for i in range(cellmask_obj.shape[0])]

#     stats_tab = pd.DataFrame({'organelle':organelle_name,
#                             'mask':'cell',
#                             'n_z':cellmask_obj.shape[0],
#                             'z':[z_bins],
#                             'z_cm_vox_cnt':[cellmask_proj.tolist()],
#                             'z_org_vox_cnt':[org_proj.tolist()],
#                             'z_org_intensity':[img_proj.tolist()],
#                             'z_nuc_vox_cnt':[nucleus_proj.tolist()]})
#     return stats_tab
    

# Zernicke routines.  inspired by cellprofiler, but heavily simplified
### USED ###
def zernike_metrics(pixels,z):
    """
    
    """
    vr = np.sum(pixels[:,:,np.newaxis]*z.real, axis=(0,1))
    vi = np.sum(pixels[:,:,np.newaxis]*z.imag, axis=(0,1))    
    magnitude = np.sqrt(vr * vr + vi * vi) / pixels.sum()
    phase = np.arctan2(vr, vi)
    # return {"zer_mag": magnitude, "zer_phs": phase}
    return magnitude, phase


## USED ###
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
    

   
# def get_zernike_metrics(        
#         cellmask_proj: np.ndarray,
#         nucleus_proj: Union[np.ndarray, None], 
#         org_proj: np.ndarray,
#         organelle_name: str,
#         zernike_degree: int = 9 
#         ):

#     """
    
#     """

#     labels = label(cellmask_proj>0) #extent as 0,1 rather than bool
#     zernike_indexes = centrosome.zernike.get_zernike_indexes( zernike_degree + 1)


#     z = zernike_polynomial(labels, zernike_indexes)

#     z_cm = zernike_metrics(cellmask_proj, z)
#     z_org = zernike_metrics(org_proj, z)
#     z_nuc = zernike_metrics(nucleus_proj, z)


#     # nm_labels = [f"{n}_{m}" for (n, m) in (zernike_indexes)
#     stats_tab = pd.DataFrame({'organelle':organelle_name,
#                                 'mask':'cell',
#                                 'zernike_n':[zernike_indexes[:,0].tolist()],
#                                 'zernike_m':[zernike_indexes[:,1].tolist()],
#                                 'zernike_cm_mag':[z_cm[0].tolist()],
#                                 'zernike_cm_phs':[z_cm[1].tolist()],   
#                                 'zernike_obj_mag':[z_org[0].tolist()],
#                                 'zernike_obj_phs':[z_org[1].tolist()],
#                                 'zernike_nuc_mag':[z_nuc[0].tolist()],
#                                 'zernike_nuc_phs':[z_nuc[1].tolist()]})

#     return stats_tab

### USED ###
def get_zernike_metrics(        
        cellmask_proj: np.ndarray,
        nucleus_proj: Union[np.ndarray, None], 
        org_proj: np.ndarray,
        organelle_name: str,
        zernike_degree: int = 9 ):

    """
    
    """

    labels = label(cellmask_proj>0) #extent as 0,1 rather than bool
    zernike_indexes = centrosome.zernike.get_zernike_indexes( zernike_degree + 1)


    z = zernike_polynomial(labels, zernike_indexes)

    z_cm = zernike_metrics(cellmask_proj, z)
    z_org = zernike_metrics(org_proj, z)
    z_nuc = zernike_metrics(nucleus_proj, z)



    # nm_labels = [f"{n}_{m}" for (n, m) in (zernike_indexes)
    stats_tab = pd.DataFrame({'object':organelle_name,
                                'zernike_n':[zernike_indexes[:,0].tolist()],
                                'zernike_m':[zernike_indexes[:,1].tolist()],
                                'zernike_mask_mag':[z_cm[0].tolist()],
                                'zernike_mask_phs':[z_cm[1].tolist()],   
                                'zernike_obj_mag':[z_org[0].tolist()],
                                'zernike_obj_phs':[z_org[1].tolist()],
                                'zernike_center_mag':[z_nuc[0].tolist()],
                                'zernike_center_phs':[z_nuc[1].tolist()]})

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

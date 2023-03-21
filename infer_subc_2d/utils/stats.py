import numpy as np
from skimage.measure import regionprops_table, regionprops, mesh_surface_area, marching_cubes, label
from skimage.morphology import binary_erosion
from skimage.measure._regionprops import _props_to_dict
from typing import Tuple, Any

from infer_subc_2d.utils.img import apply_mask

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


def get_summary_stats_3D(input_obj: np.ndarray, intensity_img, mask: np.ndarray) -> Tuple[Any, Any]:
    """collect volumentric stats from skimage.measure.regionprops
        properties = ["label","max_intensity", "mean_intensity", "min_intensity" ,"area"->"volume" , "equivalent_diameter",
        "centroid", "bbox","euler_number", "extent"
        +   extra_properties = [standard_deviation_intensity]

    Parameters
    ------------
    input_obj:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    soma_mask:
        a 3d image containing the soma object (mask)
    mask:
        a 3d image containing the soma object (mask)

    Returns
    -------------
    pandas dataframe of stats and the regionprops object
    """

    # mask
    # intensity_img = apply_mask(intensity_img,mask )  #not needed
    input_obj = apply_mask(input_obj, mask)
    labels = label(input_obj).astype("int")
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

    rp = regionprops(labels, intensity_image=intensity_img, extra_properties=extra_properties)

    props = _my_props_to_dict(
        rp, labels, intensity_image=intensity_img, properties=properties, extra_properties=extra_properties
    )
    # props = regionprops_table(
    #     labels, intensity_image=intensity_img, properties=properties, extra_properties=extra_properties
    # )
    props["surface_area"] = surface_area_from_props(labels, props)
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


def get_simple_stats_3D(A, mask):
    """collect volumentric stats of A"""

    properties = ["label"]  # our index to organelles
    # add area
    properties = properties + ["area", "equivalent_diameter"]
    #  position:
    properties = properties + ["centroid", "bbox"]  # ,  'weighted_centroid']
    # etc
    properties = properties + ["slice"]

    labels = label(apply_mask(A, mask)).astype("int")
    # props = regionprops_table(labels, intensity_image=None,
    #                             properties=properties, extra_properties=[])

    rp = regionprops(labels, intensity_image=None, extra_properties=[])
    props = _my_props_to_dict(rp, labels, intensity_image=None, properties=properties, extra_properties=None)

    stats_table = pd.DataFrame(props)
    stats_table.rename(columns={"area": "volume"}, inplace=True)

    return stats_table, rp


def get_AintB_stats_3D(A, B, mask, shell_A=False):
    """collect volumentric stats of A intersect B"""
    properties = ["label"]  # our index to organelles
    # add area
    properties = properties + ["area", "equivalent_diameter"]
    #  position:
    properties = properties + ["centroid", "bbox"]  # ,  'weighted_centroid']
    # etc
    properties = properties + ["slice"]

    Alab = label(A).astype("int")
    Blab = label(B).astype("int")

    if shell_A:
        A = np.logical_xor(A, binary_erosion(A))

    A_int_B = np.logical_and(A, B)

    labels = label(apply_mask(A_int_B, mask)).astype("int")

    props = regionprops_table(labels, intensity_image=None, properties=properties, extra_properties=None)

    props["surface_area"] = surface_area_from_props(labels, props)
    props["label_A"] = [Alab[s].max() for s in props["slice"]]
    props["label_B"] = [Blab[s].max() for s in props["slice"]]
    props_table = pd.DataFrame(props)
    props_table.rename(columns={"area": "volume"}, inplace=True)
    props_table.drop(columns="slice", inplace=True)

    return props_table


# untested 2D version
def get_summary_stats_2D(input_obj, intensity_img, mask):
    """collect volumentric stats"""

    # mask
    # intensity_img = apply_mask(intensity_img,mask )  #not needed
    input_obj = apply_mask(input_obj, mask)
    labels = label(input_obj).astype("int")
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

    rp = regionprops(labels, intensity_image=intensity_img, extra_properties=extra_properties)
    props = _my_props_to_dict(
        rp, labels, intensity_image=intensity_img, properties=properties, extra_properties=extra_properties
    )
    props_table = pd.DataFrame(props)
    #  # ETC.  skeletonize via cellprofiler /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/morphologicalskeleton.py
    #             y_data = skimage.morphology.skeletonize(x_data)
    # /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/measureobjectskeleton.py
    return props_table, rp

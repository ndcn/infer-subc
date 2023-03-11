import numpy as np
from skimage.measure import regionprops_table, regionprops, mesh_surface_area, marching_cubes, label

from infer_subc_2d.utils.img import apply_mask

import pandas as pd


def get_summary_stats_3D(input_obj, intensity_img, mask):
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
    properties = properties + ["centroid", "bbox"]  # , 'bbox', 'weighted_centroid']
    # etc
    properties = properties + ["euler_number", "extent"]  # only works for BIG organelles: 'convex_area','solidity',

    props = regionprops_table(
        labels, intensity_image=intensity_img, properties=properties, extra_properties=extra_properties
    )

    props["surface_area"] = surface_area_from_props(labels, props)

    stats_table = pd.DataFrame(props)
    stats_table.rename({"area": "volume"})

    #  # ETC.  skeletonize via cellprofiler /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/morphologicalskeleton.py
    #         if x.volumetric:
    #             y_data = skimage.morphology.skeletonize_3d(x_data)
    # /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/measureobjectskeleton.py

    rp = regionprops(labels, intensity_image=intensity_img, extra_properties=extra_properties)
    return stats_table, rp


def surface_area_from_props(labels, props):
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

    table = regionprops_table(
        labels, intensity_image=intensity_img, properties=properties, extra_properties=extra_properties
    )
    stats_table = pd.DataFrame(table)

    #  # ETC.  skeletonize via cellprofiler /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/morphologicalskeleton.py
    #             y_data = skimage.morphology.skeletonize(x_data)
    # /Users/ahenrie/Projects/Imaging/CellProfiler/cellprofiler/modules/measureobjectskeleton.py

    rp = regionprops(labels, intensity_image=intensity_img, extra_properties=extra_properties)
    return stats_table, table, rp

import numpy as np

from platform import system

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NewType,
    Sequence,
    Tuple,
    Type,
    Union,
)
from collections import defaultdict

import dask.array
import zarr
import xarray as xr

from aicsimageio.writers import OmeTiffWriter

# This is a WOEFULLY inadequate stub for a duck-array type.
# Mostly, just a placeholder for the concept of needing an ArrayLike type.
# Ultimately, this should come from https://github.com/napari/image-types
# and should probably be replaced by a typing.Protocol
# note, numpy.typing.ArrayLike (in v1.20) is not quite what we want either,
# since it includes all valid arguments for np.array() ( int, float, str...)
ArrayLike = Union[np.ndarray, "dask.array.Array", "zarr.Array"]
# layer data may be: (data,) (data, meta), or (data, meta, layer_type)
# using "Any" for the data type until ArrayLike is more mature.
FullLayerData = Tuple[Any, Dict, str]
LayerData = Union[Tuple[Any], Tuple[Any, Dict], FullLayerData]

PathLike = Union[str, Path]
PathOrPaths = Union[str, Sequence[str]]

from logging import getLogger

logger = getLogger(__name__)
###############################################################################

AICSIMAGEIO_CHOICES = "AICSImageIO Scene Management"
CLEAR_LAYERS_ON_SELECT = "Clear All Layers on New Scene Selection"
UNPACK_CHANNELS_TO_LAYERS = "Unpack Channels as Layers"

SCENE_LABEL_DELIMITER = " :: "

# Threshold above which to use out-of-memory loading
IN_MEM_THRESHOLD_PERCENT = 0.3
IN_MEM_THRESHOLD_SIZE_BYTES = 4e9  # 4GB
###############################################################################

from aicsimageio import AICSImage, exceptions
from aicsimageio.dimensions import DimensionNames

from dataclasses import dataclass


@dataclass
class AICSImageReaderWrap:
    """
    Simple dataclass wrapper for the AICSImage output to prepare for imprting to our bioim class
    TODO: make a nice reppr
    """

    name: str
    image: xr.DataArray
    meta: Dict[str, Any]
    raw_meta: Tuple[Dict[str, Any], Union[Dict[str, Any], List]]

    def __init__(self, name: str, image: xr.DataArray, meta: Dict[str, Any]):
        self.name = name
        self.image = image
        self.meta = meta
        self.raw_meta = get_raw_meta_data(meta)


def reader_function(path: "PathLike", in_memory: Union[bool, None] = None) -> Union[List["LayerData"], None]:
    """
    Given a single path return a list of LayerData tuples.
    """
    # Only support single path
    if isinstance(path, list):
        logger.info("AICSImageIO: Multi-file reading not yet supported.")
        return None

    if in_memory is None:
        from aicsimageio.utils.io_utils import pathlike_to_fs
        from psutil import virtual_memory

        fs, path = pathlike_to_fs(path)
        imsize = fs.size(path)
        available_mem = virtual_memory().available
        _in_memory = imsize <= IN_MEM_THRESHOLD_SIZE_BYTES and imsize / available_mem <= IN_MEM_THRESHOLD_PERCENT
    else:
        _in_memory = in_memory

    # Alert console of how we are loading the image
    logger.info(f"AICSImageIO: Reader will load image in-memory: {_in_memory}")

    # Open file and get data
    img = AICSImage(path)

    # Check for multiple scenes
    if len(img.scenes) > 1:
        logger.info(
            f"AICSImageIO: Image contains {len(img.scenes)} scenes. "
            f"Supporting more than the first scene is experimental. "
            f"Select a scene from the list widget. There may be dragons!"
        )
        # # Launch the list widget
        # _get_scenes(path=path, img=img, in_memory=_in_memory)

        # Return an empty LayerData list; ImgLayers will be handled via the widget.
        # HT Jonas Windhager
        return [(None,)]
    else:
        data = _get_full_image_data(img, in_memory=_in_memory)
        meta = _get_meta(data, img)
        return [(data, meta, "image")]


def _get_full_image_data(
    img: AICSImage,
    in_memory: bool,
) -> xr.DataArray:
    if DimensionNames.MosaicTile in img.reader.dims.order:
        try:
            if in_memory:
                return img.reader.mosaic_xarray_data.squeeze()

            return img.reader.mosaic_xarray_dask_data.squeeze()

        # Catch reader does not support tile stitching
        except NotImplementedError:
            logger.warning("AICSImageIO: Mosaic tile stitching " "not yet supported for this file format reader.")

    if in_memory:
        print(f"xarray_data")
        retval = img.reader.xarray_data.squeeze()
        print(f"type(retval)={type(retval)}")
        return retval

    return img.reader.xarray_dask_data.squeeze()


# Function to get Metadata to provide with data
def _get_meta(data: xr.DataArray, img: AICSImage) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if DimensionNames.Channel in data.dims:
        # Construct basic metadata
        channels_with_scene_index = [
            f"{img.current_scene_index}{SCENE_LABEL_DELIMITER}"
            f"{img.current_scene}{SCENE_LABEL_DELIMITER}{channel_name}"
            for channel_name in data.coords[DimensionNames.Channel].data.tolist()
        ]
        meta["name"] = channels_with_scene_index
        meta["channel_axis"] = data.dims.index(DimensionNames.Channel)

    # Not multi-channel, use current scene as image name
    else:
        meta["name"] = img.reader.current_scene

    # Handle samples / RGB
    if DimensionNames.Samples in img.reader.dims.order:
        meta["rgb"] = True

    # Handle scales
    scale: List[float] = []
    for dim in img.reader.dims.order:
        if dim in [
            DimensionNames.SpatialX,
            DimensionNames.SpatialY,
            DimensionNames.SpatialZ,
        ]:
            scale_val = getattr(img.physical_pixel_sizes, dim)
            if scale_val is not None:
                scale.append(scale_val)

    # Apply scales
    if len(scale) > 0:
        meta["scale"] = tuple(scale)

    # Apply all other metadata
    img_meta = {"aicsimage": img, "raw_image_metadata": img.metadata}
    try:
        img_meta["ome_types"] = img.ome_metadata
    except Exception:
        pass

    meta["metadata"] = img_meta
    return meta


def export_ome_tiff(data_in, meta_in, img_name, out_path, curr_chan=0) -> str:
    #  data_in: types.ArrayLike,
    #  meta_in: dict,
    # img_name: types.PathLike,
    # out_path: types.PathLike,
    # curr_chan: int
    # assumes a single image

    out_name = out_path + img_name + ".ome.tiff"

    image_names = [img_name]
    print(image_names)
    # chan_names = meta_in['metadata']['aicsimage'].channel_names

    physical_pixel_sizes = [meta_in["metadata"]["aicsimage"].physical_pixel_sizes]

    dimension_order = ["CZYX"]
    channel_names = [meta_in["metadata"]["aicsimage"].channel_names[curr_chan]]
    if len(data_in.shape) == 3:  # single channel zstack
        data_in = data_in[np.newaxis, :, :, :]

    if data_in.dtype == "bool":
        data_in = data_in.astype(np.uint8)
        data_in[data_in > 0] = 255

    out_ome = OmeTiffWriter.build_ome(
        [data_in.shape],
        [data_in.dtype],
        channel_names=channel_names,  # type: ignore
        image_name=image_names,
        physical_pixel_sizes=physical_pixel_sizes,
        dimension_order=dimension_order,
    )

    OmeTiffWriter.save(
        data_in,
        out_name,
        dim_order=dimension_order,
        channel_names=channel_names,
        image_names=image_names,
        physical_pixel_sizes=physical_pixel_sizes,
        ome_xml=out_ome,
    )
    return out_name


### UTILS
def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]["#text"] = text
        else:
            d[t.tag] = text
    return d


def get_raw_meta_data(meta_dict):
    curr_platform = system()

    if curr_platform == "Linux":
        raw_meta_data = meta_dict["metadata"]["raw_image_metadata"].dict()
        ome_types = meta_dict["metadata"]["ome_types"]
    elif curr_platform == "Darwin":
        raw_meta_data = meta_dict["metadata"]["raw_image_metadata"]
        ome_types = []
    else:
        raw_meta_data = meta_dict["metadata"]["raw_image_metadata"]
        ome_types = []
        print(f"warning: platform = '{curr_platform}' is untested")
    return (raw_meta_data, ome_types)


def read_input_image(image_name):
    # from aicsimageio import AICSImage
    # from napari_aicsimageio.core import _get_full_image_data, _get_meta
    # img_in = AICSImage(czi_image_name)
    # data_out = _get_full_image_data(img_in, in_memory=True)
    # meta_out = _get_meta(data, img_in)
    # meta_out['AICSImage'] = img_in

    # prefer this wrapper because it returns numpy arrays
    # or more simply with napari_aicsimagie io
    data_out, meta_out, layer_type = reader_function(image_name)[0]
    return AICSImageReaderWrap(image_name, data_out, meta_out)

import numpy as np

from platform import system
# import os
# import pickle

from pathlib import Path
from collections import defaultdict

from typing import Dict, Union, List, Any, Tuple

from aicsimageio.writers import OmeTiffWriter

# from napari_aicsimageio.core import reader_function
from aicsimageio import AICSImage, exceptions

# import ome_types
from tifffile import imwrite, tiffcomment, imread
import time

# todo depricate wrapper
from dataclasses import dataclass

from infer_subc.utils._aicsimage_reader import reader_function, export_ome_tiff  # , export_tiff


# TODO:  remove reader_function overhead for writting "intermediate" .tif files


def read_ome_image(image_name):
    """
    return output from napari aiscioimage reader
    """
    data_out, meta_out, layer_type = reader_function(image_name, in_memory=True)[0]

    meta_out["file_name"] = image_name
    return (data_out, meta_out)


def read_czi_image(image_name):
    """
    return output from napari aiscioimage reader (alias for read_ome_image)
    """
    return read_ome_image(image_name)


def read_tiff_image(image_name):
    """
    return tiff image with tifffile.imread.  Using the `reader_function` (vial read_ome_image) and AICSimage is too slow
        prsumably handling the OME meta data is what is so slow.
    """
    # start = time.time()
    image = imread(
        image_name,
    )
    # end = time.time()
    # print(f">>>>>>>>>>>> tifffile.imread  (dtype={image.dtype} in ({(end - start):0.2f}) sec")
    return image  # .get_image_data("CZYX")


def export_tiff(
    data_in: np.ndarray,
    img_name: str,
    out_path: Union[Path, str],
    channel_names: Union[List[str], None] = None,
    meta_in: Union[Dict, None] = None,
) -> int:
    """
    wrapper for exporting  tiff with tifffile.imwrite
     --> usiong AICSimage is too slow
        prsumably handling the OME meta data is what is so slow.
    """

    # start = time.time()

    out_name = Path(out_path, f"{img_name}.tiff")

    # TODO: add metadata OR simpliify and pass name rather than meta-data
    # image_names = [img_name]
    # # chan_names = meta_in['metadata']['aicsimage'].channel_names
    # physical_pixel_sizes = [meta_in["metadata"]["aicsimage"].physical_pixel_sizes]
    # # dimension_order = ["CZYX"]
    # if channel_names is None:
    #     channel_names = [meta_in["metadata"]["aicsimage"].channel_names]
    # else:
    #     channel_names = [channel_names]
    # if len(data_in.shape) == 3:  # single channel zstack
    #     dimension_order = ["ZYX"]
    #     # data_in = data_in[np.newaxis, :, :, :]
    # elif len(data_in.shape) == 2:  # single channel , 1Z
    #     dimension_order = ["YX"]
    #     # data_in = data_in[np.newaxis, np.newaxis, :, :]
    #     physical_pixel_sizes[0] = [physical_pixel_sizes[0][1:]]

    dtype = data_in.dtype
    if dtype == "bool" or dtype == np.uint8:
        data_in = data_in.astype(np.uint16)
        data_in[data_in > 0] = 1
        dtype = data_in.dtype
        # print(f"changed `bool` -> {dtype}")
    # else:
        # print(f"export as {dtype}")

    ret = imwrite(
            out_name,
            data_in,
            dtype=dtype,
            # metadata={
            #     "axes": dimension_order,
            #     # "physical_pixel_sizes": physical_pixel_sizes,
            #     # "channel_names": channel_names,
            # },
        )
    # end = time.time()
    # print(f">>>>>>>>>>>> tifffile.imwrite in ({(end - start):0.2f}) sec")
    return ret


# function to collect all the
def list_image_files(data_folder: Path, file_type: str, postfix: Union[str, None] = None) -> List:
    """
    get a list of all the filetypes
    TODO: aics has cleaner functions than this "lambda"
    should this use Path methods? or return Path?
    """

    if postfix is not None:
        return sorted(data_folder.glob(f"*{postfix}{file_type}"))
    else:
        return sorted(data_folder.glob(f"*{file_type}"))

    # if prefix is not None:
    #     return [
    #         os.path.join(data_folder, f_name)
    #         for f_name in os.listdir(data_folder)
    #         if f_name.endswith(file_type) and f_name.startswith(prefix)
    #     ]
    # else:
    #     return [os.path.join(data_folder, f_name) for f_name in os.listdir(data_folder) if f_name.endswith(file_type)]


## DEPRICATE BELOW?


# TODO:  depricate AICSImageReaderWrap
def read_input_image(image_name):
    """
    send output from napari aiscioimage reader wrapped in dataclass
    """
    data_out, meta_out, layer_type = reader_function(image_name)[0]
    return AICSImageReaderWrap(image_name, data_out, meta_out)


@dataclass
class AICSImageReaderWrap:
    """
    Simple dataclass wrapper for the AICSImage output to prepare for imprting to our bioim class
    TODO: make a nice reppr
    """

    name: str
    image: np.ndarray
    meta: Dict[str, Any]
    raw_meta: Tuple[Dict[str, Any], Union[Dict[str, Any], List]]

    def __init__(self, name: str, image: np.ndarray, meta: Dict[str, Any]):
        self.name = name
        self.image = image
        self.meta = meta
        self.raw_meta = get_raw_meta_data(meta)


def import_inferred_organelle(name: str, meta_dict: Dict, out_data_path: Path, file_type: str) -> Union[np.ndarray, None]:
    """
    read inferred organelle from ome.tif file

    Parameters
    ------------
    name: str
        name of organelle.  i.e. nuc, lyso, etc.
    meta_dict:
        dictionary of meta-data (ome) from original file
    out_data_path:
        Path object of directory where tiffs are read from
    file_type: 
        The type of file you want to import as a string (ex - ".tif", ".tiff", ".czi", etc.)

    Returns
    -------------
    exported file name

    """

    img_name = Path(meta_dict["file_name"])

    if name is None:
        pass
    else:
        organelle_fname = f"{img_name.stem}-{name}{file_type}"

        organelle_path = out_data_path / organelle_fname

        if Path.exists(organelle_path):
            # organelle_obj, _meta_dict = read_ome_image(organelle_path)
            organelle_obj = read_tiff_image(organelle_path)  # .squeeze()
            print(f"loaded  inferred {len(organelle_obj.shape)}D `{name}`  from {out_data_path} ")
            return organelle_obj
        else:
            print(f"`{name}` object not found: {organelle_path}")
            raise FileNotFoundError(f"`{name}` object not found: {organelle_path}")


# TODO throw exception and call with try
# def import_inferred_organelle(name: str, meta_dict: Dict, out_data_path: Path) -> Union[np.ndarray, None]:
#     """
#     read inferred organelle from ome.tif file

#     Parameters
#     ------------
#     name: str
#         name of organelle.  i.e. nuc, lyso, etc.
#     meta_dict:
#         dictionary of meta-data (ome) from original file
#     out_data_path:
#         Path object of directory where tiffs are read from

#     Returns
#     -------------
#     exported file name

#     """

#     # copy the original file name to meta
#     img_name = Path(meta_dict["file_name"])  #
#     # add params to metadata

#     organelle_fname = f"{img_name.stem}-{name}.tiff"

#     organelle_path = out_data_path / organelle_fname

#     if Path.exists(organelle_path):
#         # organelle_obj, _meta_dict = read_ome_image(organelle_path)
#         organelle_obj = read_tiff_image(organelle_path)  # .squeeze()
#         print(f"loaded  inferred {len(organelle_obj.shape)}D `{name}`  from {out_data_path} ")
#         return organelle_obj
#     else:
#         print(f"`{name}` object not found: {organelle_path}")
#         raise FileNotFoundError(f"`{name}` object not found: {organelle_path}")


def export_inferred_organelle(img_out: np.ndarray, name: str, meta_dict: Dict, out_data_path: Path) -> str:
    """
    write inferred organelle to ome.tif file

    Parameters
    ------------
    img_out:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    name: str
        name of organelle.  i.e. nuc, lyso, etc.
    meta_dict:
        dictionary of meta-data (ome) only using original file name here, but could add metadata
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """
    # get some top-level info about the RAW data
    # channel_names = meta_dict['name']
    # img = meta_dict['metadata']['aicsimage']
    # scale = meta_dict['scale']
    # channel_axis = meta_dict['channel_axis']

    # copy the original file name to meta
    img_name = Path(meta_dict["file_name"])  #
    # add params to metadata

    if not Path.exists(out_data_path):
        Path.mkdir(out_data_path)
        print(f"making {out_data_path}")

    img_name_out = f"{img_name.stem}-{name}"
    # HACK: skip the ome
    # out_file_n = export_ome_tiff(img_out, meta_dict, img_name_out, str(out_data_path) + "/", name)
    out_file_n = export_tiff(img_out, img_name_out, out_data_path, name, meta_dict)
    print(f"saved file: {img_name_out}")
    return out_file_n


def export_inferred_organelle_stack(img_out, layer_names, meta_dict, data_root_path):
    """
    stack all the inferred objects and stack along 0 dimension
    """
    # get some top-level info about the RAW data
    channel_names = meta_dict["name"]
    img = meta_dict["metadata"]["aicsimage"]
    scale = meta_dict["scale"]
    channel_axis = meta_dict["channel_axis"]

    img_name = Path(meta_dict["file_name"])  #
    # add params to metadata
    meta_dict["layer_names"] = layer_names
    out_path = data_root_path / "inferred_objects"
    name = "stack"
    img_name_out = f"{img_name.stem}-{name}"

    out_file_n = export_ome_tiff(img_out, meta_dict, img_name_out, str(out_path), layer_names)
    print(f"saved file: {out_file_n}")
    return out_file_n



# # DEPRICATE
# def append_ome_metadata(filename, meta_add):
#     filename = "test.ome.tif"

#     imwrite(
#         filename,
#         np.random.randint(0, 1023, (4, 256, 256, 3), "uint16"),
#         bigtiff=True,
#         photometric="RGB",
#         tile=(64, 64),
#         metadata={
#             "axes": "ZYXS",
#             "SignificantBits": 10,
#             "Plane": {"PositionZ": [0.0, 1.0, 2.0, 3.0]},
#         },
#     )

#     ome_xml = tiffcomment(filename)
#     ome = ome_types.from_xml(ome_xml)
#     ome.images[0].description = "Image 0 description"
#     ome_xml = ome.to_xml()
#     tiffcomment(filename, ome_xml)


### UTILS
def etree_to_dict(t):
    """
    etree dumper from stackoverflow use to dump meta_dict[metadata][raw_image_metadata]
    """
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
    """
    not sure why the linux backend works for ome... need to solve
    """
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





def export_ndarray(data_in, img_name, out_path) -> str:
    """
    #  data_in: types.ArrayLike,
    #  meta_in: dict,
    # img_name: types.PathLike,
    # out_path: types.PathLike,
    # curr_chan: int
    # assumes a single image
    """
    out_name = out_path + img_name + ".npy"
    data_in.tofile(out_name)
    return out_name


#  The methods below are sloooooow.  the overhead loading in image is ~5.8 seconds
#  2 seconds to read the image, compared to .01 sec for directly reading...
def import_inferred_organelle_AICS(name: str, meta_dict: Dict, out_data_path: Path) -> Union[np.ndarray, None]:
    """
    read inferred organelle from ome.tif file with AICSIMAGEIO

    Parameters
    ------------
    name: str
        name of organelle.  i.e. nuc, lyso, etc.
    meta_dict:
        dictionary of meta-data (ome) from original file
    out_data_path:
        Path object of directory where tiffs are read from

    Returns
    -------------
    exported file name

    """
    img_name = Path(meta_dict["file_name"])
    # HACK: skip OME
    # organelle_fname = f"{name}_{img_name.split('/')[-1].split('.')[0]}.ome.tiff"

    organelle_fname = f"{img_name.stem}-{name}.tiff"

    organelle_path = out_data_path / organelle_fname

    if Path.exists(organelle_path):
        # organelle_obj, _meta_dict = read_ome_image(organelle_path)
        organelle_obj = read_tiff_image_AICS(organelle_path)  # .squeeze()
        print(f"loaded  inferred {len(organelle_obj.shape)}D `{name}`  from {out_data_path} ")
        return organelle_obj > 0
    else:
        print(f"`{name}` object not found: {organelle_path}")
        raise FileNotFoundError(f"`{name}` object not found: {organelle_path}")


def export_inferred_organelle_AICS(img_out: np.ndarray, name: str, meta_dict: Dict, out_data_path: Path) -> str:
    """
    write inferred organelle to ome.tif file with AICSIMAGEIO

    Parameters
    ------------
    img_out:
        a 3d  np.ndarray image of the inferred organelle (labels or boolean)
    name: str
        name of organelle.  i.e. nuc, lyso, etc.
    meta_dict:
        dictionary of meta-data (ome)
    out_data_path:
        Path object where tiffs are written to

    Returns
    -------------
    exported file name

    """
    img_name = Path(meta_dict["file_name"])  #

    if not Path.exists(out_data_path):
        Path.mkdir(out_data_path)
        print(f"making {out_data_path}")

    img_name_out = f"{img_name.stem}-{name}"
    out_file_n = export_tiff_AICS(img_out, img_name_out, out_data_path, name, meta_dict)

    print(f"saved file: {out_file_n}")
    return out_file_n


def read_tiff_image_AICS(image_name):
    """aicssegmentation way to do it"""
    start = time.time()
    image = AICSImage(image_name)
    if len(image.scenes) > 1:
        raise ValueError("Multi-Scene images are unsupported")

    if image.dims.T > 1:
        raise ValueError("Timelapse images are unsupported.")

    if image.dims.C > 1:
        im_out = image.get_image_data("CZYX")

    im_out = image.get_image_data("ZYX")
    end = time.time()
    print(f">>>>>>>>>>>> AICSImage read  (dtype={image.dtype}in ({(end - start):0.2f}) sec")
    return im_out


def export_tiff_AICS(
    data_in: np.ndarray,
    img_name: str,
    out_path: Union[Path, str],
    channel_names: Union[List[str], None] = None,
    meta_in: Union[Dict, None] = None,
) -> str:
    """
    aicssegmentation way to do it
    """
    start = time.time()
    # img_name = meta_in["file_name"]  #
    # add params to metadata
    out_name = Path(out_path, img_name + ".tiff")

    if data_in.dtype == "bool":
        data_in = data_in.astype(np.uint8)
        data_in[data_in > 0] = 255

    OmeTiffWriter.save(data=data_in, uri=out_name.as_uri(), dim_order="ZYX")
    end = time.time()
    print(f">>>>>>>>>>>> export_tiff_AICS ({(end - start):0.2f}) sec")
    print(f"saved file AICS {out_name}")
    return out_name

    # OmeTiffWriter.save(data=self._format_output(result), uri=output_path, dim_order="ZYX")

    # DEPRICATED OMETIFFWRITER CODE
    # dimension_order = ["CZYX"]
    # if len(data_in.shape) == 3:  # single channel zstack
    #     data_in = data_in[np.newaxis, :, :, :]
    # elif len(data_in.shape) == 2:  # single channel , 1Z
    #     dimension_order = ["YX"]
    #     data_in = data_in[np.newaxis, np.newaxis, :, :]
    #     physical_pixel_sizes = [meta_in["metadata"]["aicsimage"].physical_pixel_sizes]

    # out_ome = OmeTiffWriter.build_ome(
    #     [data_in.shape],
    #     [data_in.dtype],
    #     channel_names=channel_names,  # type: ignore
    #     image_name=image_names,
    #     physical_pixel_sizes=physical_pixel_sizes,
    #     dimension_order=dimension_order,
    # )

    # OmeTiffWriter.save(
    #     data_in,
    #     out_name,
    #     dim_order=dimension_order,
    #     channel_names=channel_names,
    #     image_names=image_names,
    #     physical_pixel_sizes=physical_pixel_sizes,
    #     ome_xml=None,
    # )
    # return out_name

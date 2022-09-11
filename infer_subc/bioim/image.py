from pathlib import Path

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, NewType, Sequence, Tuple, Type, Union, Optional
import dask.array as da

import math

from aicsimageio.dimensions import Dimensions

# derived from CellProfiler/core
from infer_subc.imports import *
from infer_subc.bioim.base import BioImContainer, BioImLike
from infer_subc.transforms.transform import BioImTransform
from infer_subc.transforms.pipeline import BioImPipeline

from ..utils.file_io import AICSImageReaderWrap, PathLike
import xarray as xr

# called "BioImage to disambiguate from other Image classes which are running around"
class BioImImage(BioImContainer):
    """
    channel - the channel containing the structure we want to extract.  e.g. which "organelle"
    image - a BioImLike array.  numpy, xarray, or dask.  requires other propert

    mask - a binary image indicating the points of interest in the image. The
    mask is the same size as the image.
    parent_image - for derived images, the parent that was used to create this
    image. This image may inherit attributes from the parent image, such as
    the masks used to create the parent
    # easy access attributes
    path_name - the path name to the file holding the image or None for a
    derived image
    file_name - the file name of the file holding the image or None for a
    derived image
    scale - the scaling suggested by the initial image format (e.g., 4095 for
    a 12-bit a/d converter).

    """

    def __init__(self, im: AICSImageReaderWrap = None, channel: int = None, mask: BioImLike = None):
        """ """
        # otherwise load the image and infer everything.
        if im is not None:
            image = im.image
            meta = im.meta
            fname = im.name
            raw_meta, ome_types = im.raw_meta

            self.ndim = image.ndim
            # TODO: safety logic
            # if (image.dims[0] == 'C') and
            self.set_image(image, channel)
            self.file_name = fname
            self.metadata = meta

        else:
            self._image = None
            self.file_name = None
            self.metadata = None

        if mask is not None:
            self._mask = mask
            self._has_mask = True
        else:
            self._mask = None
            self._has_mask = False

        self.channel = channel
        self.__transforms = []

    def get_image(self):
        """Return the primary image"""
        return self._image

    def set_image(self, img: BioImLike, channel=None, convert=True):
        """Set the primary image
        Convert the image to a np array of dtype = np.float64.
        Rescale according to Matlab's rules for im2double:
        * single/double values: keep the same
        * uint8/16/32/64: scale 0 to max to 0 to 1
        * int8/16/32/64: scale min to max to 0 to 1
        * logical: save as is (and get if must_be_binary)
        TODO:  automatically convert BioImLike input to internal xr.DataArray
            # use scale etc to create the coords / dims
        """
        # Assert that we are about to have a 3D image by choosing channel
        # right now assume C Z X Y
        if channel is not None:
            _img = img[channel, :, :, :].squeeze()
            self.ndim -= 1

        # convert dask/xarray to numpy
        img = np.asanyarray(_img)

        if img.dtype.name == "bool" or not convert:
            self._image = img
            return

        mval = 0.0
        scale = 1.0
        fix_range = False
        if issubclass(img.dtype.type, np.floating):
            pass
        elif img.dtype.type is np.uint8:
            scale = math.pow(2.0, 8.0) - 1
        elif img.dtype.type is np.uint16:
            scale = math.pow(2.0, 16.0) - 1
        elif img.dtype.type is np.uint32:
            scale = math.pow(2.0, 32.0) - 1
        elif img.dtype.type is np.uint64:
            scale = math.pow(2.0, 64.0) - 1
        elif img.dtype.type is np.int8:
            scale = math.pow(2.0, 8.0)
            mval = -scale / 2.0
            scale -= 1
            fix_range = True
        elif img.dtype.type is np.int16:
            scale = math.pow(2.0, 16.0)
            mval = -scale / 2.0
            scale -= 1
            fix_range = True
        elif img.dtype.type is np.int32:
            scale = math.pow(2.0, 32.0)
            mval = -scale / 2.0
            scale -= 1
            fix_range = True
        elif img.dtype.type is np.int64:
            scale = math.pow(2.0, 64.0)
            mval = -scale / 2.0
            scale -= 1
            fix_range = True
        # Avoid temporaries by doing the shift/scale in place.
        img = img.astype(np.float32)
        img -= mval
        img /= scale
        if fix_range:
            # These types will always have ranges between 0 and 1. Make it so.
            np.clip(img, 0, 1, out=img)

        self.scale = scale  # output dtype is np,float32
        _img.data = img
        self._image = _img

    image = property(get_image, set_image)

    def set_transforms(self, transforms):
        if isinstance(transforms, BioImPipeline):
            self.__transforms = transforms.transforms

        elif isinstance(transforms, List):
            assert transforms is None or all([isinstance(t, BioImTransform) for t in transforms]), (
                f"All elements in input list must be of" f" type pathml.core.Transform"
            )
            self.__transforms = transforms

        elif isinstance(transforms, BioImTransform):
            self.__transforms = [transforms]

    def get_transforms(self):
        return self.__transforms

    def get_pipeline(self):
        return BioImPipeline(self.__transforms)

    def add_transform(self, new_transform: BioImTransform):
        transforms = self.__transforms
        transforms.append(new_transform)
        self.__transforms = transforms

    # def metadata(self) -> Any:
    @property
    def dtype(self) -> np.dtype:
        return self._image.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._image.shape

    # def dims(self) -> Dimensions:
    #  def pipeline(self) -> BioImPipeline:
    #  def mask(self) -> Any:
    # def channel_names(self) -> Optional[List[str]]:
    # def physical_pixel_sizes(self) -> PhysicalPixelSizes:

    # @property
    # def multichannel(self):
    #     return True if self._image.ndim == self.dimensions + 1 else False

    @property
    def volumetric(self):
        return self.ndim == 3

    @property
    def dask_data(self) -> da.Array:
        return da.from_array(self.image)

    @property
    def xarray_data(self) -> xr.DataArray:
        return self.image

    # @abstractmethod
    @property
    def data(self) -> np.ndarray:
        return self.image.data

    # @abstractmethod
    @data.setter
    def data(self, img):
        self._image = xr.DataArray(img, coords=self._image.coords, dims=self._image.dims)

    @property
    def mask(self):
        """Return the mask (pixels to be considered) for the primary image"""
        if self._mask is not None:
            return self._mask
        # default to ones_like image... but no over channel.
        shape = self.image.shape
        # if self.multichannel:
        #     shape = shape[-self.dimensions :]

        return np.ones(shape, dtype=bool)

    @mask.setter
    def mask(self, mask):
        """Set the mask (pixels to be considered) for the primary image
        Convert the input into a np array. If the input is numeric,
        we convert it to boolean by testing each element for non-zero.
        """
        m = np.array(mask)

        if not (m.dtype.type is bool):
            m = m != 0
        self._mask = m

    @property
    def has_mask(self) -> bool:
        """True if the image has a mask"""
        return self.mask is not None

    @property
    def file_name(self) -> PathLike:
        """The name of the file holding this image"""
        return self._file_name

    @file_name.setter
    def file_name(self, fname: PathLike):
        """The name of the file holding this image"""
        self._file_name = fname

    @property
    def scale(self):
        """The scale at acquisition
        This is the intensity scale used by the acquisition device. For
        instance, a microscope might use a 12-bit a/d converter to acquire
        an image and store that information using the TIF MaxSampleValue
        tag = 4095.
        """
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale

    @property
    def ndim(self) -> int:
        """
        number of active dimensions
        """
        return self._ndim

    @ndim.setter
    def ndim(self, ndim: int):
        self._ndim = ndim

    @property
    def metadata(self) -> Dict:
        """
        number of active dimensions
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict):
        self._metadata = metadata

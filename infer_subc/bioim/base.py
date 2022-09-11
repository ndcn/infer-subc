from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Union, Protocol

from abc import ABC, abstractmethod

from aicsimageio import AICSImage

from ..imports import *

###############################################################################
###############################################################################

# IO Types
ArrayLike = Union[np.ndarray, da.Array]
MetaArrayLike = Union[ArrayLike, xr.DataArray]
BioImLike = Union[ArrayLike, MetaArrayLike, List[MetaArrayLike]]


# Image Utility Types
class PhysicalPixelSizes(NamedTuple):
    Z: Optional[float]
    Y: Optional[float]
    X: Optional[float]


class BioImContainer(ABC):
    @abstractmethod
    def get_image(self) -> xr.DataArray:
        pass

    @abstractmethod
    def set_image(self, image: xr.DataArray):  # MetaArrayLike
        pass

    @property
    def xarray_data(self) -> xr.DataArray:
        pass

    @property
    def dask_data(self) -> da.Array:
        pass

    @property
    def data(self) -> np.ndarray:
        pass

    @property
    def metadata(self) -> Any:
        pass

    @property
    def dtype(self) -> np.dtype:
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        pass

    @property
    def ndim(self) -> int:
        pass

    @property
    def mask(self) -> Any:
        pass

    @property
    def transforms(self) -> Any:
        pass

    @abstractmethod
    def add_transform(self, transform: Any):  # MetaArrayLike
        pass

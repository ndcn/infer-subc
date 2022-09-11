import pickle

from typing import List
from abc import ABC, abstractmethod

from .transform import BioImTransformBase
from ..utils.file_io import PathLike
from ..bioim.base import BioImContainer, MetaArrayLike

###############################################################################


class BioImPipelineBase(ABC):
    """
    Base class for all Transforms.
    Each transform must operate on a Tile.
    """

    @abstractmethod
    def save(self, image: MetaArrayLike) -> MetaArrayLike:
        ...

    @abstractmethod
    def apply(self, img: BioImContainer):
        ...


class BioImPipeline(BioImPipelineBase):
    """
    Compose a sequence of Transforms
    Args:
        transform_sequence (list): sequence of transforms to be consecutively applied.
            List of `pathml.core.Transform` objects
    """

    def __init__(self, transform_sequence: List[BioImTransformBase] = None):
        assert transform_sequence is None or all([isinstance(t, BioImTransformBase) for t in transform_sequence]), (
            f"All elements in input list must be of" f" type pathml.core.Transform"
        )
        self.transforms = transform_sequence

    def __len__(self):
        return len(self.transforms)

    def __repr__(self):
        if self.transforms is None:
            return "Pipeline()"
        else:
            out = f"Pipeline([\n"
            for t in self.transforms:
                out += f"\t{repr(t)},\n"
            out += "])"
            return out

    def apply(self, image: BioImContainer):
        # this function has side effects
        # modifies the tile in place, but also returns the modified tile
        # need to do this for dask distributed
        if self.transforms:
            for t in self.transforms:
                t.apply(image)
        return image

    def save(self, filename: PathLike):
        """
        save pipeline to disk
        Args:
            filename (str): save path on disk
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        # pickle.dump(self, open(filename, "wb"))

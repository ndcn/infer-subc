"""
Functions and classes to capture expected "norms" of the inferred organelles

"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Union, Protocol

from abc import ABC, abstractmethod

###############################################################################
###############################################################################

# IO Types
ArrayLike = Union[np.ndarray, da.Array]

# Image Utility Types
class ObjectStats(NamedTuple):
    extrema: Tuple[Any,Any]
    mu: Optional[Any]
    sigma: Optional[Any]
    skew: Optional[Any]
    kurt: Optional[Any]


class ObjectCheck(ABC):
    @property
    def prior(self) -> ObjectStats:
        pass

    @abstractmethod
    def check_prior(self, test_im: ArrayLike) -> bool:  
        pass
  
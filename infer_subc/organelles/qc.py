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
    n_organelles: int
    size_organelles: int
    total_size: Optional[Any]
    volumetric: bool 
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
  



# cellmask = ObjectCheck
# cellmask

# Expected Number of Recognized Objects at 63x/2.2x in iPSCs
# Cellmask=1 (only 1 for analysis)
# Nuclei= 7-15
# Lysosomes= 10-200
# Mitochondria= 10-100
# Golgi=1-20
# Peroxisomes=10-100
# ER= 1
# Lipid Droplets=0-20

# Expected Area (tot pixel) of Recognized Objects at 63x/2.2x in iPSCs
# Cellmask= 50,000-200,000
# Nuclei= 10,000-50,000
# Lysosomes= 1,000-10,000
# Mitochondria= 1,000-10,000
# Golgi= 800-6,000
# Peroxisomes= 500- 2,000
# ER= 3,000-20,000
# Lipid Droplets= 0-300

# Expected Size (mean pixel) of Recognized Objects at 63x/2.2x in iPSCs
# Nuclei= 10,000-50,000
# Lysosomes= 100-500
# Mitochondria= 100-500
# Golgi= 100-1000
# Peroxisomes= 20-60
# Lipid Droplets= 10-30 (when present and good de-clumping)



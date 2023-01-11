# infer_subc_2d/organelles

"""The module contains the following submodules:

- `nuclei` - infer_nuclei, fixed_infer_nuclei
- `soma` - infer_soma, fixed_infer_soma, non_linear_soma_transform_MCZ, masked_inverted_watershed, raw_soma_MCZ
- `cytosol` - infer_cytosol
- `lysosome` - infer_lysosome, fixed_infer_lysosome, lysosome_filiment_filter, lysosome_spot_filter
- `mitochondria` - infer_mitochondria, fixed_infer_mitochondria
- `golgi` - infer_golgi, fixed_infer_golgi
- `peroxisome` - infer_peroxisome, fixed_infer_peroxisome
- `er` - infer_endoplasmic_reticulum, fixed_infer_endoplasmic_reticulum
- `lipid` - infer_lipid, fixed_infer_lipid
- `zslice` - find_optimal_Z, fixed_find_optimal_Z, get_optimal_Z_image, fixed_get_optimal_Z_image

Examples:
    >>> from calculator import calculations
    >>> calculations.add(2, 4)
    6.0
    >>> calculations.multiply(2.0, 4.0)
    8.0
    >>> from calculator.calculations import divide
    >>> divide(4.0, 2)
    2.0

"""

from .nuclei import infer_nuclei, fixed_infer_nuclei
from .soma import infer_soma, fixed_infer_soma, non_linear_soma_transform_MCZ, masked_inverted_watershed, raw_soma_MCZ
from .cytosol import infer_cytosol
from .lysosome import infer_lysosome, fixed_infer_lysosome, lysosome_filiment_filter, lysosome_spot_filter
from .mitochondria import infer_mitochondria, fixed_infer_mitochondria
from .golgi import infer_golgi, fixed_infer_golgi
from .peroxisome import infer_peroxisome, fixed_infer_peroxisome
from .er import infer_endoplasmic_reticulum, fixed_infer_endoplasmic_reticulum
from .lipid import infer_lipid, fixed_infer_lipid
from .zslice import find_optimal_Z, fixed_find_optimal_Z, get_optimal_Z_image, fixed_get_optimal_Z_image

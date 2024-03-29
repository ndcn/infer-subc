
## infer_subc

This module contains code to segment organelles from multi-channel images generated by SCohenLab.   The "raw" data files are the output of linear unmixing of multi-spectral imaging capture in their lab.

NOTE:  this is designed to work with a second repo `organelle-segmenter-plugin` which instantiates a plugin for napari.

In addition to the python based module there are a series of expository Jupyter notebooks which demonstrate the logic and development of the library.

### organelles
These are function to infer each specific organelles from their respective channels: Nuclei, Cellmask (Cell Membrane TBD), Lysosome, Mitochondria, Golgi, Peroxisome, Endoplasmic Reticulum, and Lipid bodies.

### core
This submodule contains functions for handling the file systems and input / output, as well as the core image processing.  The bulk of the image processing functions are simple wrappers to `scipy` and `numpy` image processing functions as well as functions from the Allen Cell Segmentation (`aicssegmentaion`) library.  `utils.img` contains most of the specific image processing routines employed in segmentation, while `utils.file_io` handles loading and saving the data files.

### utils
This submodule contains functions for handling the file systems and input / output, as well as the core image processing.  The bulk of the image processing functions are simple wrappers to `scipy` and `numpy` image processing functions as well as functions from the Allen Cell Segmentation (`aicssegmentaion`) library.  `utils.img` contains most of the specific image processing routines employed in segmentation, while `utils.file_io` handles loading and saving the data files.

### workflows
This submodule (hard forked from `aicssegmentation`) works with the napari plugin to provide interactive GUI control of the segmentaitons.

### etc
#### batch
This submodule contains functions to process each multi-channel/spectral image to infer ALL organelles

#### constants, exceptions
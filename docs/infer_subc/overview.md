
# infer_subc

This module contains code to segment organelles from multi-channel images and quantify their morphology, interactions, and subcellular distribution.

### assets
A set of information documents available in the the [`organelle-segmenter-plugin`](/docs/infer_subc/napari/plugin.md) in Napari.

### core
Functions to process input images and handle segmentation output files.

### organelles
Functions specific to the segmentation of each specified organelle.

### organelles_config
.json files to specify the workflows available in [`organelle-segmenter-plugin`](/docs/infer_subc/napari/organelle_config.md).

### utils
Functions to quantify organelle features, etc.

### workflow
Functions to create and update workflows for [`organelle-segmenter-plugin`](/docs/infer_subc/napari/plugin.md)
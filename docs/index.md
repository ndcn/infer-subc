# Welcome to infer_subc

### `infer-subc` 
- aims to create a <ins>simple</ins>, <ins>extensible</ins>, and <ins>reproducible</ins> pipeline to segment (or infer) and quantify the shape, position, size, and interaction of multiple intracellular organelles from confocal microscopy 🔬 images. 
- is <ins>modular</ins> 🔢 to support a diversity of organelle-focused research questions. 
- can be <ins>applied broadly</ins> to many types of in vitro 🧫 and in vivo models 🐁🧬 to better understand the spatial coordination and interactome of organelles during key biological processes or disease. 

## Overview
### Instance segmentation of organelles 🖍️
The starting point for the `infer-subc` analysis pipeline is to perform instance segmentation on single or multichannel confocal microscopy images, where each channel labels a different intracellular component. Each organelle will be segmented from a single intensity channel. The subsequent analysis is performed at a single-cell level by also segmenting a mask the cell area.

**More information of the segmentation process can be found [here](/docs/segmentation.md).**

### Organelle Quantification 🧮📐
After each of the organelles of interest are segmented, single or multi-organelle analysis can be carried out using Jupyter Notebook-based pipeline(s). We incorporated quantification of the `morphology`, `interactions`, and `distribution` of any number of organelles within the same cell. Together these metrics summarize the *"Organelle Signature"* of a particular cell type or cellular condition. Examples of this method are cited below. 

**Information on the organelle quantification pipeline can be found [here](/docs/quantification.md).**

## Publications
`infer-subc` analysis has been featured in:
1. Shannon N. Rhoads, Weizhen Dong, Chih-Hsuan Hsu, Ngudiankama R. Mfulama, Joey V. Ragusa, Michael Ye, Andy Henrie, Maria Clara Zanellati, Graham H. Diering, Todd J. Cohen, Sarah Cohen. *Neurons and astrocytes have distinct organelle signatures and responses to stress.* bioRxiv 2024.10.30.621066; doi: https://doi.org/10.1101/2024.10.30.621066


## FRAMEWORK, RESOURCES & CONTRIBUTIONS

### Segmentation workflow & Napari plugin design:
Early in the develepmont we chose to leverage methods created in the `Allen Cell & Structure Segmenter` and [`napari plugin`](https://www.napari-hub.org/plugins/napari-allencell-segmenter). Although the logic of our **multi-channel** organelle segmentations required us to fork and modify their code, we hope it porvides a stable, but evolving base which will help manage accumulation of technical debt. In addition to the overall logic, we particulary leverage their *workflow* paradigm which is integral in the use of the napari plugin interface. Implementation of `infer-subc` as a Napari plugin using this framework is called [`organelle-segmenter-plugin`](https://github.com/ndcn/organelle-segmenter-plugin).

​The `Allen Cell & Structure Segmenter` is a Python-based open source toolkit developed at the Allen Institute for Cell Science for 3D segmentation of intracellular structures in fluorescence microscope images: [`aicssegmentation`](https://github.com/AllenCell/aics-segmentation) package. This toolkit brings together classic image segmentation and iterative deep learning workflows first to generate initial high-quality 3D intracellular structure segmentations and then to easily curate these results to generate the ground truths for building robust and accurate deep learning models. The toolkit takes advantage of the high replicate 3D live cell image data collected at the Allen Institute for Cell Science of over 30 endogenous fluorescently tagged human induced pluripotent stem cell (hiPSC) lines. Each cell line represents a different intracellular structure with one or more distinct localization patterns within undifferentiated hiPS cells and hiPSC-derived cardiomyocytes. Here, we leveraged select segmentation methods specialized for a particular organelle shape (i.e., round versus tubular objects) to carried out segmentation states within the [`infer-subc` segmentation workflows](/docs/segmentation.md).

### Support of this project includes:
- [CZI Neurodegeneration Challenge Network (NDCN)](https://chanzuckerberg.com/science/programs-resources/neurodegeneration-challenge/)
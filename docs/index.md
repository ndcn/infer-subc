# Welcome to infer_subc

 `infer_subc` aims to create a simple and extensible workflow of image analysis leveraging [scipy image](link), and [napari](link) for reproducable analysis with an intuitive interface. 

This is a simple repo to collect code and documentations from the pilot project kicking off as part of the CZI Neurodegeneration Challenge Network [(NDCN)](https://chanzuckerberg.com/science/programs-resources/neurodegeneration-challenge/) Data Science Concierge program.  The PILOT study is a collaboration with Cohen lab at UNC [(website,](https://cohenlaboratory.web.unc.edu/) [github)](https://github.com/SCohenLab) to migrate a multispectral imaging dataset of iPSCs which identifies sub-cellular components to a scalable cloud-based pipeline.  

## Overview
In general we ware interested in making segmentations which are the inferred organelles for each cell.   Procedurally we will need to define some masking ROIs which correspond to the cellmask, wholecell (cellmask + dendrites) and cytoplasm (cellmask minus nucleus.)  THerefore we need to first infer nuclei, cellmask, and cytoplasm as the first steps.  

From there the organelle segmentations are largely independent, and we will employ the cytoplasm (or cellmask for the nucleus segmentation) as a mask when collecting statistics about the individual organelle objects. 

Notebooks  found [here]( link ) provide the template

### Sub-Cellular object Inference PIPELINE OVERVIEW

#### GOAL:  Infer sub-cellular components in order to understand interactome 

To measure shape, position, size, and interaction of eight organelles/cellular components (Nuclei (NU), Lysosomes (LS),Mitochondria (MT), Golgi (GL), Peroxisomes (PO), Endoplasmic Reticulum (ER), Lipid Droplet (LD), and CELLMASK) during differentiation of iPSCs, in order to understand the Interactome / Spatiotemporal coordination.


#### summary of _OBJECTIVES_ ✅
- robust inference of subcellular objects:
  - Nescessary masks and nuclei
    -  #### 1️⃣. [infer NUCLEI ](./infer_subc/organelles.md)
    -  #### 2️⃣. [Infer CELLMASK](./infer_subc/organelles.md) (🚨🚨🚨🚨 Steps 2-9 depend on establishing a good solution here.)
    -  #### 3️⃣. [Infer CYTOPLASM](./infer_subc/organelles.md) 

  - individual organelles
    -  #### 4️⃣. [Infer LYSOSOMES](./infer_subc/organelles.md) 
    -  #### 5️⃣. [Infer MITOCHONDRIA](./infer_subc/organelles.md)
    -  #### 6️⃣. [Infer GOLGI complex](./infer_subc/organelles.md)
    -  #### 7️⃣. [Infer PEROXISOMES](./infer_subc/organelles.md)
    -  #### 8️⃣. [Infer ENDOPLASMIC RETICULUM ](./infer_subc/organelles.md)
    -  #### 9️⃣. [Infer LB](./nbs/overview.md) 


----------------------------
## FRAMEWORKS & RESOURCES

### NOTE: PIPELINE TOOL AND DESIGN CHOICES?
Early in the develepmont we chose to leverage the Allen Cell & Structure Segmenter and [napari plugin](https://www.napari-hub.org/plugins/napari-allencell-segmenter).   Although the logic of our **multi-channel** organelle segmentations required us to fork and modify their code, we hope it porvides a stable, but evolving base which will help manage accumulationof technical debt.   In addition to the overall logic, we particulary leverage their `workflow` paradigm which is crucial for leveraging the napari plugin interface of widgets to the segmentation workflows.


#### ​The Allen Cell & Structure Segmenter 
​The Allen Cell & Structure Segmenter is a Python-based open source toolkit developed at the Allen Institute for Cell Science for 3D segmentation of intracellular structures in fluorescence microscope images: `aicssegmentation` [package](https://github.com/AllenCell/aics-segmentation).  This toolkit brings together classic image segmentation and iterative deep learning workflows first to generate initial high-quality 3D intracellular structure segmentations and then to easily curate these results to generate the ground truths for building robust and accurate deep learning models. The toolkit takes advantage of the high replicate 3D live cell image data collected at the Allen Institute for Cell Science of over 30 endogenous fluorescently tagged human induced pluripotent stem cell (hiPSC) lines. Each cell line represents a different intracellular structure with one or more distinct localization patterns within undifferentiated hiPS cells and hiPSC-derived cardiomyocytes.

-------

For full documentation visit [github.io.com/infer_subc](https://ndcn.github.io/infer-subc/).
Robust inference of subcellular objects:

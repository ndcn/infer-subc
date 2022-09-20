
---
# infer_subc

[![codecov](https://codecov.io/gh/ergonyc/infer-subc/branch/main/graph/badge.svg?token=infer-subc_token_here)](https://codecov.io/gh/ergonyc/infer-subc)
[![CI](https://github.com/ergonyc/infer-subc/actions/workflows/main.yml/badge.svg)](https://github.com/ergonyc/infer-subc/actions/workflows/main.yml)

 `infer_subc` aims to create a simple and extensible workflow of image analysis leveraging [scipy image](link), and [napari](link) for reproducable analysis with an intuitive interface. 

This is a simple repo to collect code and documentations from the pilot project kicking off as part of the CZI Neurodegeneration Challenge Network [(NDCN)](https://chanzuckerberg.com/science/programs-resources/neurodegeneration-challenge/) Data Science Concierge program.  The PILOT study is a collaboration with Cohen lab at UNC [(website,](https://cohenlaboratory.web.unc.edu/) [github)](https://github.com/SCohenLab) to migrate a multispectral imaging dataset of iPSCs which identifies sub-cellular components to a scalable cloud-based pipeline.   

--------------

## Overview

Notebooks  found [here]( link ) provide the template

### Sub-Cellular object Inference PIPELINE OVERVIEW

#### GOAL:  Infer sub-cellular components in order to understand interactome 

To measure shape, position, size, and interaction of eight organelles/cellular components (Nuclei (NU), Lysosomes (LS),Mitochondria (MT), Golgi (GL), Peroxisomes (PO), Endoplasmic Reticulum (ER), Lipid Droplet (LD), and SOMA) during differentiation of iPSCs, in order to understand the Interactome / Spatiotemporal coordination.

#### summary of _OBJECTIVES_ ✅
- Infer subcellular objects:
  -  #### 1️⃣. [infer NUCLEI ](./notebooks/01_infer_nuclei.ipynb)
  -  #### 2️⃣. [Infer SOMA](./notebooks/02_infer_soma.ipynb) 🚧 WIP 🚧 
  -  #### 3️⃣. [Infer CYTOSOL](./notebooks/03_infer_cytosol.ipynb) 🚧 WIP 🚧 (🚨🚨🚨🚨 Steps 3-9 depend on establishing a good solution here.)
  -  #### 4️⃣. [Infer LYSOSOMES](./notebooks/04_infer_lysosome.ipynb) 🚧 WIP 🚧
  -  #### 5️⃣. [Infer MITOCHONDRIA](./notebooks/02_infer_soma.ipynb)
  -  #### 6️⃣. [Infer GOLGI complex](./notebooks/02_infer_soma.ipynb)
  -  #### 7️⃣. [Infer PEROXISOMES](./notebooks/02_infer_soma.ipynb)
  -  #### 8️⃣. [Infer ENDOPLASMIC RETICULUM ](./notebooks/02_infer_soma.ipynb)
  -   #### 9️⃣. [Infer LB](./notebooks/02_infer_soma.ipynb) 🚧 WIP 🚧



----------------------------
## DEVELOPMENT STATUS
### DONE
- Infer subcellular objects notebooks 
  -  ### 0. [pipeline Overview/setup  ](./notebooks/00_pipeline_setup.ipynb)
  -  ### 1. [infer NUCLEI ](./notebooks/01_infer_nuclei.ipynb) 
  -  ### 2. [Infer SOMA](./notebooks/03_infer_soma.ipynb)
  -  ### 3. [Infer CYTOSOL](./notebooks/03_infer_cytosol.ipynb)
  -  ### 4. [Infer LYSOSOMES](./notebooks/04_infer_lysosome.ipynb)
  -  ### 5. [Infer MITOCHONDRIA](./notebooks/05_infer_mitochondria.ipynb)
  -  ### 6. [Infer GOLGI complex](./notebooks/06_infer_golgi.ipynb)
  -  ### 7. [Infer PEROXISOMES](./notebooks/07_infer_peroxisome.ipynb)
  -  ### 8. [Infer ENDOPLASMIC RETICULUM ](./notebooks/08_infer_endoplasmic_reticulum.ipynb)

## WIP
- Infer subcellular objects:
  -  ### 9. [Infer LB](./notebooks/09_infer_lipid_bodies.ipynb)
- Infer subcellular objects:
  -  ### 10. [batch process](./notebooks/10_batch_process.ipynb)

- leverage [aics-segmenter]( https://allencell.org/segmenter ) `Workflows` paradigm (from [`aicssegmentation.workflow`](https://github.com/AllenCell/aics-segmentation) encode procedures as .json and execute programmically.
- develop a [napari](https://napari.org/stable/) plugin: [napari-infer-subc](https://github.com/ergonyc/napari-infer-subc) 
### TO-DO
- QC for each step, each organelle
- testing
- TBD

## FRAMEWORKS & RESOURCES

### NOTE: PIPELINE TOOL AND DESIGN CHOICES?
We want to leverage the Allen Cell & Structure Setmenter.  It has been wrapped as a [napari-plugin](https://www.napari-hub.org/plugins/napari-allencell-segmenter) but fore the workflow we are proving out here we will want to call the `aicssegmentation` [package](https://github.com/AllenCell/aics-segmentation) directly.

#### ​The Allen Cell & Structure Segmenter 
​The Allen Cell & Structure Segmenter is a Python-based open source toolkit developed at the Allen Institute for Cell Science for 3D segmentation of intracellular structures in fluorescence microscope images. This toolkit brings together classic image segmentation and iterative deep learning workflows first to generate initial high-quality 3D intracellular structure segmentations and then to easily curate these results to generate the ground truths for building robust and accurate deep learning models. The toolkit takes advantage of the high replicate 3D live cell image data collected at the Allen Institute for Cell Science of over 30 endogenous fluorescently tagged human induced pluripotent stem cell (hiPSC) lines. Each cell line represents a different intracellular structure with one or more distinct localization patterns within undifferentiated hiPS cells and hiPSC-derived cardiomyocytes.

More details about Segmenter can be found at https://allencell.org/segmenter
In order to leverage the A
### IMPORTS
- `napari` for visualization
- `scipy`, `ndimage`, and `skimage` for image analysis (also `itk` occationally)
-  `numpy` `ndarrays` under the hood


### `napari` 
**napari** is a fast, interactive, multi-dimensional image viewer for Python. It's designed for browsing, annotating, and analyzing large multi-dimensional images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the scientific Python stack (numpy, scipy). It can be installed via python tools (i.e. `pip` or `conda`) or as a stand-alone gui.  
More info can be found on their [website](https://napari.org/stable/) and at [this repository](https://github.com/napari/napari).

A powerful extension framework of **napari plugins**  extend `napari`s functionality.   More infor can be foudn at the [napari hub](https://www.napari-hub.org/about) The napari hub seeks to solve many of the challenges and needs in finding analysis solutions to bioimaging problems. 

## ADWB hints
The medium term goal for this project is to execute it on ADDI's ADWB

[uploading guide ](https://knowledgebase.aridhia.io/article/guidance-for-uploading-files/)
[uploading files via the workspace article](https://knowledgebase.aridhia.io/article/uploading-files-via-the-workspace/).
[Using BLOB storage](https://knowledgebase.aridhia.io/article/using-blob-storage/)

### Uploading files to Blobs
> The file upload to Blob storage follows the process described in [uploading files via the workspace article](https://knowledgebase.aridhia.io/article/uploading-files-via-the-workspace/). Note that due to the nature of Blob storage, folder hierarchies cannot exist without content. This means that you won't be able to create empty folders, and after refreshing the page the empty folders will be gone from your Blob storage. There is a workaround: you can create an empty folder, and without closing the window, add or upload a new file to the folder.


## Install it from PyPI

```bash
pip install infer_subc
```

## Usage

```py
from infer_subc import BaseClass
from infer_subc import base_function

BaseClass().base_method()
base_function()
```

```bash
$ python -m infer_subc
#or
$ infer_subc
```

## Development
Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

The roadmap for this project includes extending to 
### napari plugins 
> insert link here


### templates 

---
# infer_subc

[![codecov](https://codecov.io/gh/ergonyc/infer-subc/branch/main/graph/badge.svg?token=infer-subc_token_here)](https://codecov.io/gh/ergonyc/infer-subc)
[![CI](https://github.com/ergonyc/infer-subc/actions/workflows/main.yml/badge.svg)](https://github.com/ergonyc/infer-subc/actions/workflows/main.yml)

Awesome infer_subc created by ergonyc to create a simple and extensible workflow of image analysis leveraging [scipy image](link), and [napari](link) for reproducable analysis with an intuitive interface. 

This is a simple repo to collect code and documentations from the pilot project kicking off as part of the CZI Neurodegeneration Challenge Network [(NDCN)](https://chanzuckerberg.com/science/programs-resources/neurodegeneration-challenge/) Data Science Concierge program.  The PILOT study is a collaboration with Cohen lab at UNC [(website,](https://cohenlaboratory.web.unc.edu/) [github)](https://github.com/SCohenLab) to migrate a multispectral imaging dataset of iPSCs which identifies sub-cellular components to a scalable cloud-based pipeline.   

--------------

## Overview

Notebooks  found [here]( link ) provide the template

## Sub-Cellular object Inference PIPELINE OVERVIEW

### GOAL:  Infer sub-cellular components in order to understand interactome 

To measure shape, position, size, and interaction of eight organelles/cellular components (Nuclei (NU), Lysosomes (LS),Mitochondria (MT), Golgi (GL), Peroxisomes (PO), Endoplasmic Reticulum (ER), Lipid Droplet (LD), and SOMA) during differentiation of iPSCs, in order to understand the Interactome / Spatiotemporal coordination.

### summary of _OBJECTIVES_
- Infer subcellular objects:
  -  #### #1. [infer NUCLEI ](#image-processing-objective-1-infer-nucleii) - NU, ch 0
  -  #### #2. [Infer SOMA](#image-processing-objective-2-infer-soma) (extent of single brightest cell)- SO, composite icluding ch 1, ch 4,ch 5, and ch 7 "residuals"
  -  #### #3. [Infer CYTOSOL](#image-processing-objective-3-infer-cytosol)- CY 
  -  #### #4. [Infer LYSOSOMES](#image-processing-objective-4-infer-lysosomes)  - LS, ch 1
  -  #### #5. [Infer MITOCHONDRIA](#image-processing-objective-5-infer-mitochondria) - MT, ch 2
  -  #### #6. [Infer GOLGI complex](#image-processing-objective-6-infer-golgi-complex) - GL, ch 3
  -  #### #7. [Infer PEROXISOMES](#image-processing-objective-7-infer-peroxisomes) - PO, ch  4
  -  #### #8. [Infer ENDOPLASMIC RETICULUM ](#image-processing-objective-8-infer-endoplasmic-reticulum)- ER, ch 5
  -  #### #9. [Infer LB](#image-processing-objective-9-infer-lipid-bodies-droplet), LD, ch 6 
  -  

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

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
  -  #### #1. [infer NUCLEI ](../tree/main/notebooks/02_infer_soma.ipynb)
  -  #### #2. [Infer SOMA](../tree/main/notebooks/02_infer_soma.ipynb)
  -  #### #3. [Infer CYTOSOL](../tree/main/notebooks/02_infer_soma.ipynb)
  -  #### #4. [Infer LYSOSOMES](../tree/main/notebooks/02_infer_soma.ipynb)
  -  #### #5. [Infer MITOCHONDRIA](../tree/main/notebooks/02_infer_soma.ipynb)
  -  #### #6. [Infer GOLGI complex](../tree/main/notebooks/02_infer_soma.ipynb)
  -  #### #7. [Infer PEROXISOMES](../tree/main/notebooks/02_infer_soma.ipynb)
  -  #### #8. [Infer ENDOPLASMIC RETICULUM ](../tree/main/notebooks/02_infer_soma.ipynb)
  -   #### #9. [Infer LB](../tree/main/notebooks/02_infer_soma.ipynb)




## DONE
- Infer subcellular objects:
  -  ### 0. [pipeline Overview/setup  ](../tree/main/notebooks/00_pipeline_setup.ipynb)
  -  ### 1. [infer NUCLEI ](../tree/main/notebooks/01_infer_nuclei.ipynb) 


## WIP
- Infer subcellular objects:
  -  ### 2. [Infer SOMA](../tree/main/notebooks/02_infer_soma.ipynb)
## TO-DO
- Infer subcellular objects:

  -  ### 3. [Infer CYTOSOL](../tree/main/notebooks/02_infer_soma.ipynb)
  -  ### 4. [Infer LYSOSOMES](../tree/main/notebooks/02_infer_soma.ipynb)
  -  ### 5. [Infer MITOCHONDRIA](../tree/main/notebooks/02_infer_soma.ipynb)
  -  ### 6. [Infer GOLGI complex](../tree/main/notebooks/02_infer_soma.ipynb)
  -  ### 7. [Infer PEROXISOMES](../tree/main/notebooks/02_infer_soma.ipynb)
  -  ### 8. [Infer ENDOPLASMIC RETICULUM ](../tree/main/notebooks/02_infer_soma.ipynb)
  -  ### 9. [Infer LB](../tree/main/notebooks/02_infer_soma.ipynb)



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
# infer_subc

[![codecov](https://codecov.io/gh/ergonyc/infer-subc/branch/main/graph/badge.svg?token=infer-subc_token_here)](https://codecov.io/gh/ergonyc/infer-subc)
[![CI](https://github.com/ergonyc/infer-subc/actions/workflows/main.yml/badge.svg)](https://github.com/ergonyc/infer-subc/actions/workflows/main.yml)

## About The Project

`infer_subc` 
- aims to create a simple, extensible, and reproducible workflow to measure (or infer) the shape, position, size, and interaction of several sub-cellular components. These data can then be applied later to better understand the spatial coordination of these structures and the interactome during key biological processes.

- is part of a larger collaboration between the CZI Neurodegeneration Challenge Network [(NDCN)](https://chanzuckerberg.com/science/programs-resources/neurodegeneration-challenge/) Data Science Concierge program and the Cohen lab at UNC [(website,](https://cohenlaboratory.web.unc.edu/) [github)](https://github.com/SCohenLab) to migrate a multispectral imaging dataset of iPSCs which identifies sub-cellular components to a scalable cloud-based pipeline.  



## `infer_subc` Workflow

The staring point of this workflow is a set of multichannel images, where each channel labels a different sub-cellular component. The workflow can then be completed in a _**suggested**_ series of steps, outlined in the notebooks below.

**Identify a single cell of interest**

1. [Infer cellmask](./notebooks/01_infer_cellmask.ipynb) (🚨 Steps 2-9 depend on establishing a good solution here)
2. [Infer nuclei ](./notebooks/02_infer_nuclei.ipynb)
3. [Infer cytoplasm](./notebooks/03_infer_cytoplasm.ipynb) 

**Segment each of the organelles**

4. [Infer lysosomes](./notebooks/04_infer_lysosome.ipynb)
5. [Infer mitochondria](./notebooks/05_infer_mitochondria.ipynb)
6. [Infer golgi complex](./notebooks/06_infer_golgi.ipynb)
7. [Infer peroxisomes](./notebooks/07_infer_peroxisome.ipynb)
8. [Infer endoplasmic reticulum](./notebooks/08_infer_endoplasmic_reticulum.ipynb)
9. [Infer lipid bodies](./notebooks/09_infer_lipid_body.ipynb) 

## Built With

A quick note on tools and resources used.

- [`napari-allencell-segmenter`](https://github.com/AllenCell/napari-allencell-segmenter) -- We are leveraging the framework of the `napari-allencell-segmenter` plugin, which enables powerful 3D image segmentation while taking advantage of the `napari` graphical user interface. 
- [`aicssegmentation`](https://github.com/AllenCell/aics-segmentation) -- We call the `aicssegmentation` package directly.
- [`napari`](https://napari.org/stable/) -- Used as the visualization framework, a fast, interactive, multi-domensional image viewer for Python.
- [`scipy`](https://scipy.org/install/) -- Image analysis
- [`scikit-image`](https://scikit-image.org/) -- Image analysis
- [`itk`](https://itkpythonpackage.readthedocs.io/en/master/Quick_start_guide.html) -- Image analysis
- [`numpy`](https://numpy.org/) -- Under the hood computation
- [`Alzheimer's Disease AD Workbench`](https://www.alzheimersdata.org/ad-workbench) -- We initially wanted to use the ADDI's ADWB as a method of data sharing and to serve as a computational resource.


## Getting Started

### Prerequisites



### Installation
`infer_subc` is  available on `PyPI` via: 

```
pip install infer_subc
```

If there are issues more details can be fouund in the [documentation](https://ndcn.github.io/infer-subc/config/)


## Usage - quick start
Its recommended that you use this repo along with the [`organelle-segmenter-plugin`](https://github.com/ndcn/organelle-segmenter-plugin) as in [Option A](#option-a-napari-organelle-segmenter-plugin) below.  Alternatively using the module functions directly as in [Option B](#option-b-python-script-or-notebook) would work just fine.


### Option A: Napari `organelle-segmenter-plugin`

1. Open a file in napari by dragging multi-channel .czi file onto napari which will import a multi-channel, multi-Z 'layer'. (Using the menu's defaults to `aicsIMAGEIO` reader which automatically splits mutliple channels into individual layers.  The plugin is able to support multi-dimensional data in .tiff, .tif. ome.tif, .ome.tiff, .czi)
2. Start the plugin (open napari, go to "Plugins" --> "organelle-segmenter-plugin" --> "workflow editor")
3. Select the image and channel to work on
4. Select a workflow based on the example image and target segmentation based on user's data. Ideally, it is recommend to start with the example with very similar morphology as user's data.
5. Click "Run All" to execute the whole workflow on the sample data.
6. Adjust the parameters of steps, based on the intermediate results.  A complete list of all functions can be found [here](https://github.com/ndcn/infer-subc/blob/main/infer_subc/organelles_config/function_params.md)🚧 WIP 🚧
7. Click "Run All" again after adjusting the parameters and repeat step 6 and 7 until the result is satisfactory.
8. Save the workflow
9. Close the plugin and open the **batch processing** part by (go to "Plugins" --> "organelle-segmenter-plugin" --> "batch processing")
10. Load the customized workflow saved above 
11. Load the folder with all the images to process
12. Click "Run"
13. Follow the [examples](https://github.com/ndcn/infer-subc/blob/main/notebooks/14_final_workflow.ipynb) in the `infer_subc` [repo](https://github.com/ndcn/infer-subc/) for postprocessing of the saved segmentations and generating the statistics.  

### Option B: python script or notebook 

A variety of example [notebooks](https://github.com/ndcn/infer-subc/blob/main/notebooks/) demonstrating how to use the are available in the repo.  Additional information can be found at https://ndcn.github.io/infer-subc/nbs/overview/.  


## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License
Distributed under the terms of the [BSD-3] license,
"organelle-segmenter-plugin" is free and open source software

## Issues

If you encounter any problems, please file an issue with a detailed description.

## ADWB hints

Given that the github repos are not yet whitelisted, the source directory needs to be zipped and uploaded in order to make an "editable" pip install.

[uploading guide ](https://knowledgebase.aridhia.io/article/guidance-for-uploading-files/)
[uploading files via the workspace article](https://knowledgebase.aridhia.io/article/uploading-files-via-the-workspace/).
[Using BLOB storage](https://knowledgebase.aridhia.io/article/using-blob-storage/)

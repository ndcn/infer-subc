# infer-subc
### A python-based image analysis tool to segment and quantify the morphology, interactions, and distribution of organelles.

<img src="infer_subc\assets\README.png" width="800">
<p>

# 📒 About this project

### `infer-subc` 
- aims to create a <ins>simple</ins>, <ins>extensible</ins>, and <ins>reproducible</ins> pipeline to segment (or infer) and quantify the shape, position, size, and interaction of multiple intracellular organelles from confocal microscopy 🔬 images. 
- is <ins>modular</ins> 🔢 to support a diversity of organelle-focused research questions. 
- can be <ins>applied broadly</ins> to many types of in vitro 🧫 and in vivo models 🐁🧬 to better understand the spatial coordination and interactome of organelles during key biological processes or disease. 

> ### Compatable Organelles 🔓🗝️
> 
> - `Lysosomes`
> - `Mitochondria`
> - `Golgi`
> - `Peroxisomes`
> - `Endoplasmic reticulum` 
> - `Lipid droplets`
> - `Cell`
> - `Nucleus`
>
>
>  *Outside segmentation methods can also be used to incorporate additional oragnelles.*


# Getting Started
## Setup ⚙️
`infer-subc` and the related `organelle-segmenter-plugin` for [Napari]() are available on `PyPI` via: 

```
pip install infer-subc
pip install organelle-segmenter-plugin
```

A full list of dependencies and recommended setup steps are included in [env_create.sh](./env_create.sh).

## Segmentation Workflows 🖍️

> ***NOTE**: Proceed to the Organelle Quantification section below if you have already created instance segmentations using a separate method.*

The starting point for `infer-subc` analysis pipeline is to perform instance segmentation on single or multichannel confocal microscopy images, where each channel labels a different intracellular component. Each channel or organelle will be segmented from a single intensity channel. The subsequent analysis is performed at a single-cell level, so a mask of the cell area will also be carried out.

We recommend that you use our `infer-subc` implementation for Napari called [`organelle-segmenter-plugin`](https://github.com/ndcn/organelle-segmenter-plugin). This will allow users to systematically test segmentation settings for each organelles, then batch process all organelles of interest across multiple cells at a time within the Napari GUI. Alternatively, you can utilize the included set of Jupter Notebooks or your own python script to work through the segmentation process step by step for each cell.

### <ins>Option A:</ins> [Napari Plugin](https://github.com/ndcn/organelle-segmenter-plugin) 🔌

You must have installed `organelle-segmenter-plugin` as described above to use this method. The order in which you run the workflows is not important, but for down stream quantification, you must include at least one organelle and the cell mask.

1. Open Napari. Then drag-and-drop or use the `File` > `Open File(s)...` controls to open a single- or multi-channel confocal microscopy image. This image will be used to test the segmentation settings you want to use during batch processing.
2. Start the plugin by navigating to `Plugin` > `Infer sub-Cellular Object Npe2 plugin` > `Workflow editor`. A right side panel will appear.
3. In the Workflow editor, select the image to work on. 
4. Select the workflow corresponding to your desired organelles/object. The order of channels/objects does not matter.
5. Adjust the parameters for each step in order based on the intermediate results. If you need to return to a previous step, you must restart the workflow by pressing `Close Workflow` at the bottom of the panel.
6. Save the workflow settings you like by using the `Save Workflow` option at the bottom of the panel. IMPORTANT: the file name should end with the same name as the workflow you are working on (e.g., 20241031_lysosomes.json can be saved from the 0.2.lysosome workflow). Save each of the settings files into the same location for batch processing.
7. Once all of the settings are saved, open the batch processor by going to `Plugins` > `Infer sub-Cellular Object Npe2 plugin` > `Batch processing`.
8. Load the saved workflow settings for all channels and specify the input (intensity images) and output (desired location for segmentation files) folders.
9. Click `Run`. A progress bar will allow you track your processing.


### <ins>Option B:</ins> [Notebooks](./notebooks) 📚

**Identify a single cell of interest:**

1. Infer `cellmask` and `nuclei` from an image with:
    - [nuclei labels; one or more cells per field of view (FOV)](./notebooks/part_1_segmentation_workflows/01_infer_masks_from-composite_with_nuc.ipynb)
    - [no cell or nuclei labels; one cell per field of view FOV](./notebooks/part_1_segmentation_workflows/01a_infer_masks_from-composite_single_cell.ipynb)
    - [no cell or nuclei labels; multiple cells per FOV](./notebooks/part_1_segmentation_workflows/01b_infer_masks_from-composite_multiple-cells.ipynb) 

**Segment each of the organelles:**

2. Infer [`lysosomes`](./notebooks\part_1_segmentation_workflows\02_infer_lysosome.ipynb)
3. Infer [`mitochondria`](./notebooks\part_1_segmentation_workflows\03_infer_mitochondria.ipynb)
4. Infer [`golgi`](./notebooks\part_1_segmentation_workflows\04_infer_golgi.ipynb)
5. Infer [`peroxisomes`](./notebooks\part_1_segmentation_workflows\05_infer_peroxisome.ipynb)
6. Infer [`endoplasmic reticulum (ER)`](./notebooks\part_1_segmentation_workflows\06_infer_ER.ipynb)
7. Infer [`lipid droplets`](./notebooks\part_1_segmentation_workflows\07_infer_lipid_droplet.ipynb)

### <ins>Quality Check:</ins> [Validate segmentation results]()🔎
After batch processing, we recommend you quality check your segmentation results by visually inspecting the images. The [Segmentation Validation Notebook]() is available to streamline the validation process.

🚧 *In a future verions, this notebook will also include quality checks for assumptions made during quantification (i.e., only one nucleus and ER per cell, etc.).*

## Organelle Quantification 🧮📐

After each of the organelles of interest are segmented, single or multi-organelle analysis can be carried out using Jupyter Notebook-based pipeline(s). Each of the following analysis types are modular and can be used in combination or separately.

**Combined Analysis:** 
- [__________]() quantification of the `morphology`, `interactions`, and `distribution` of any number of organelles within the same cell.

**Individual analysis pipelines :**

The following notebooks include steps to quantify features of single organelles or paris of organelles (interactions) from individual cells. These notebooks also act as a step-by-step guide to understanding each measurement type. 
- [Organelle morphology](./notebooks\part_2_quantification\1.1_organelle_morphology.ipynb)
- [Pairwise organelle interactions](./notebooks\part_2_quantification\1.2_organelle_interactions.ipynb)
- [Subcellular distribution](./notebooks\part_2_quantification\1.3_distribution_measurements.ipynb)
- [Cell/nucleus morphology](./notebooks\part_2_quantification\1.4_cell_region_morphology.ipynb)
- COMBINED ANAYSIS ONLY: [batch processing](./notebooks\part_2_quantification\1.5_combined_and_batch_processing.ipynb)
- COMBINED ANAYSIS ONLY: [per-cell summary](./notebooks\part_2_quantification\1.6_summary_stats.ipynb)

🚧 *Future implimentations of the notebooks will include batch processing capabilities (e.g., multiple cells, multiple organelles).*

# Additional Information
## Built With
A quick note on tools and resources used.

- [`napari-allencell-segmenter`](https://github.com/AllenCell/napari-allencell-segmenter) -- We are leveraging the framework of the `napari-allencell-segmenter` plugin, which enables powerful 3D image segmentation while taking advantage of the `napari` graphical user interface. 
- [`aicssegmentation`](https://github.com/AllenCell/aics-segmentation) -- We call the `aicssegmentation` package directly to access their advanced segmentation functions.
- [`napari`](https://napari.org/stable/) -- Used as the visualization framework, a fast, interactive, multi-domensional image viewer for Python.
- [`scipy`](https://scipy.org/install/) -- Image analysis
- [`scikit-image`](https://scikit-image.org/) -- Image analysis
- [`itk`](https://itkpythonpackage.readthedocs.io/en/master/Quick_start_guide.html) -- Image analysis
- [`numpy`](https://numpy.org/) -- Under the hood computation


## Issues
If you encounter any problems, please file an issue with a detailed description.

## Development
Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License
Distributed under the terms of the [BSD-3] license,
"organelle-segmenter-plugin" is free and open source software

## Support of this project includes:
- [CZI Neurodegeneration Challenge Network (NDCN)](https://chanzuckerberg.com/science/programs-resources/neurodegeneration-challenge/)

# Publications

`infer-subc` analysis has been featured in:
1. Shannon N. Rhoads, Weizhen Dong, Chih-Hsuan Hsu, Ngudiankama R. Mfulama, Joey V. Ragusa, Michael Ye, Andy Henrie, Maria Clara Zanellati, Graham H. Diering, Todd J. Cohen, Sarah Cohen. *Neurons and astrocytes have distinct organelle signatures and responses to stress.* bioRxiv 2024.10.30.621066; doi: https://doi.org/10.1101/2024.10.30.621066
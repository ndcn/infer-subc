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

The starting point for the `infer-subc` analysis pipeline is to perform instance segmentation on single or multichannel confocal microscopy images, where each channel labels a different intracellular component. Each organelle will be segmented from a single intensity channel. The subsequent analysis is performed at a single-cell level, so we have also developed several workflows to segement the cell area.

We recommend that you use our `infer-subc` implementation for Napari called [`organelle-segmenter-plugin`](https://github.com/ndcn/organelle-segmenter-plugin) for segmentation. This will allow users to systematically test segmentation settings for each organelles, then batch process all organelles of interest across multiple cells. Alternatively, you can utilize the included set of Jupter Notebooks or your own python script calling functions from the infer_subc module to work through the segmentation process step by step for each cell.

### <ins>Option A:</ins> [Napari Plugin](https://github.com/ndcn/organelle-segmenter-plugin) 🔌

You must have installed `organelle-segmenter-plugin` as described above to use this method. The order in which you run the workflows is not important, but for down stream quantification, you must include at least one organelle and the cell mask.

1. Open Napari. Then drag-and-drop or use the `File` > `Open File(s)...` controls to open a single- or multi-channel confocal microscopy image. This image will be used to test the segmentation settings you want to apply to many cells during batch processing.
2. Start the plugin by navigating to `Plugin` > `Infer sub-Cellular Object Npe2 plugin` > `Workflow editor`. A right side panel will appear.
3. In the Workflow editor, select the image to work on. 
4. Select the workflow corresponding to your desired organelle(s)/masks.
5. Adjust the parameters for each step based on the intermediate results. If you need to return to a previous step, you must restart the workflow by pressing `Close Workflow` at the bottom of the panel.
6. Save the workflow settings that work for your image by using the `Save Workflow` option at the bottom of the panel. Save each of the settings files to process a single image into the same location for batch processing. *IMPORTANT: the file name should end with the same name as the workflow you are working on.*
    > 
    > <ins>**Naming Examples**</ins>: 
    >
    > For settings saved from the 0.2.lyso workflow, the following naming are **acceptable**:
    > - "20241031_lyso.json"
    > - "iPSCs_lyso.json"
    > - "lyso.json"
    > 
    > Do **NOT** use names like:
    > - "lysosomes.json"
    > - "LS.json"
7. Once all of the settings are saved, open the batch processor by going to `Plugins` > `Infer sub-Cellular Object Npe2 plugin` > `Batch processing`.
8. Load the saved workflow settings for all channels and specify the input (intensity images) and output (desired location for segmentation files) folders.
9. Click `Run`. A progress bar will allow you track your processing.

> **Using sample data in Napari**
> 
> If you would like to test out the Napari plugin using the **sample data**, first download the raw astrocyte and neuron image from bioimage archive. To download the two raw microscopy images you may click [neuron](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_deconvolution/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome.tiff) and [astrocyte](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_deconvolution/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome.tiff), refer to the [Sample Data Info document](sample_data\sample_data_info.md) or follow the instructions in the beginning of [notebook 1.0](notebooks/part_1_segmentation_workflows/1.0_image_setup.ipynb). Back in Napari, Drag-and-drop or use the `File` > `Open File(s)...` controls to open one of the raw sample images. As stated above in step 2, back in Napari, open the plugin by navigating to `Plugin` > `Infer sub-Cellular Object Npe2 plugin` > `Workflow editor`. After selecting the image in the plugin, click the space next to `Add Workflow` and navigate to the `sample_data` folder inside infer_subc. For either the `example_astrocyte` or `example_neuron` folders, you will find a `settings` subfolder that contains parameters in the form of .JSON files preset to work with their matching sample images. Select the .JSON file corresponding to your desired segmentation (make sure to use the settings that match the cell type of the example image). As you would in step 4, select the workflow that appears at the bottom of the list to apply the preset parameters. You can then follow steps 5 through 9 as stated above to create the segmentations. For further elaboration on the sample data, click **[here](sample_data\sample_data_info.md)**.


### <ins>Option B:</ins> [Notebooks](/docs/nbs/overview.md) 📚
The primary purpose of the Juptyer notebooks are to walk step-by-step through each of the segementation workflows. We hope these notebooks provide a more easily accessible resource for those who are new to Python image analysis. The notebooks are built to run the segmentations using sample data if the appropriate setup is followed in [notebook 1.0](notebooks/part_1_segmentation_workflows/1.0_image_setup.ipynb). More information about the sample data is provided **[here](sample_data\sample_data_info.md)**.
*The notebooks below include steps to segment a single image. A batch processing workflow has not yet been created.*

**Step 1️⃣: Identify a single cell of interest**

Segment (or infer) the `cellmask` and `nuclei` from an image depending on the type of fluorescent labels used and the number of cells per field of view (FOV):

- Fluorescently labeled nuclei (no cell or plasma membrane)
    - [one or more cells per FOV](/notebooks/part_1_segmentation_workflows/1.1_infer_masks_from-composite_with_nuc.ipynb)
- No cell, plasma membrane, or nuclei labels with
    - [one cell per field of view FOV](/notebooks/part_1_segmentation_workflows/1.1a_infer_masks_from-composite_single_cell.ipynb)
    - [one or more cells per FOV](/notebooks/part_1_segmentation_workflows/1.1b_infer_masks_from-composite_multiple-cells.ipynb)

**Step 2️⃣: Segment organelles**

Each of the organelles you wish to include in your analysis should be segmented from a single fluorescently labeled structure.

2. Infer [`lysosomes`](/notebooks/part_1_segmentation_workflows/1.2_infer_lysosome.ipynb)
3. Infer [`mitochondria`](/notebooks/part_1_segmentation_workflows/1.3_infer_mitochondria.ipynb)
4. Infer [`golgi`](/notebooks/part_1_segmentation_workflows/1.4_infer_golgi.ipynb)
5. Infer [`peroxisomes`](/notebooks/part_1_segmentation_workflows/1.5_infer_peroxisome.ipynb)
6. Infer [`endoplasmic reticulum (ER)`](/notebooks/part_1_segmentation_workflows/1.6_infer_ER.ipynb)
7. Infer [`lipid droplets`](/notebooks/part_1_segmentation_workflows/1.7_infer_lipid_droplet.ipynb)

### <ins>Quality Check:</ins> [Validate segmentation results]()🔎
After processing all cells in your dataset, we recommend you quality check your segmentation results by visually inspecting the images. The Segmentation Validation pipeline is included in the [Full Quantification Pipeline Notebook](/notebooks/part_2_quantification/full_quantification_pipeline.ipynb) to streamline the validation process.

🚧 *In a future verions, this notebook will also include quality checks for assumptions made during quantification (i.e., only one nucleus and ER per cell, etc.).*

## Organelle Quantification 🧮📐

After each of the organelles of interest are segmented, single or multi-organelle analysis can be carried out using Jupyter Notebook-based pipeline(s). Each of the following analysis types are modular and can be used in combination or separately. The notebooks are built to run the quantification pipelines using sample data if the appropriate setup is followed in [notebook 1.0](notebooks/part_1_segmentation_workflows/1.0_image_setup.ipynb) and the segmentations the sample data is segmented using part 1 of infer-subc. More information about the sample data is provided **[here](sample_data\sample_data_info.md)**.

**Combined Analysis:** 
- [Full Quantification Pipeline](./notebooks\part_2_quantification\full_quantification_pipeline.ipynb):  quantification of the `morphology`, `interactions`, and `distribution` of any number of organelles within the same cell. This pipeline incorporates batch processing within a single experiment and summarizes the results across multiple experimental replicates.

**Individual analysis pipelines :**

The following notebooks primarily act as a step-by-step guide to understanding each measurement type. However, they can also be used to quantify features of single organelles or pairs of organelles (interactions) from individual cells.  
- [Organelle morphology](./notebooks\part_2_quantification\1.1_organelle_morphology.ipynb)
- [Pairwise organelle interactions](./notebooks\part_2_quantification\1.2_organelle_interactions.ipynb)
- [Subcellular distribution](./notebooks\part_2_quantification\1.3_distribution_measurements.ipynb)
- [Cell/nucleus morphology](./notebooks\part_2_quantification\1.4_cell_region_morphology.ipynb)
- COMBINED ANAYSIS ONLY: [batch processing](./notebooks\part_2_quantification\1.5_combined_and_batch_processing.ipynb)
- COMBINED ANAYSIS ONLY: [per-cell summary](./notebooks\part_2_quantification\1.6_summary_stats.ipynb)

🚧 *Future implimentations of these notebooks will include batch processing capabilities (e.g., multiple cells, multiple organelles) for each quantification type separately.*

# Additional Information
## Built With
A quick note on tools and resources used...

- [`napari-allencell-segmenter`](https://github.com/AllenCell/napari-allencell-segmenter) -- We are leveraging the framework of the `napari-allencell-segmenter` plugin, which enables powerful 3D image segmentation while taking advantage of the `napari` graphical user interface. 
- [`aicssegmentation`](https://github.com/AllenCell/aics-segmentation) -- We call the `aicssegmentation` package directly to access their advanced segmentation functions.
- [`napari`](https://napari.org/stable/) -- Used as the visualization framework, a fast, interactive, multi-domensional image viewer for Python.
- [`scipy`](https://scipy.org/install/) -- Image analysis
- [`scikit-image`](https://scikit-image.org/) -- Image analysis
- [`itk`](https://itkpythonpackage.readthedocs.io/en/master/Quick_start_guide.html) -- Image analysis
- [`numpy`](https://numpy.org/) -- Under the hood computation
- ['pandas'](https://pandas.pydata.org/) -- Quantitative data manipulation

### Segmentation workflow & Napari plugin design:
Early in the develepmont we chose to leverage methods created in the `Allen Cell & Structure Segmenter` and [`napari plugin`](https://www.napari-hub.org/plugins/napari-allencell-segmenter). Although the logic of our **multi-channel** organelle segmentations required us to fork and modify their code, we hope it porvides a stable, but evolving base which will help manage accumulation of technical debt. In addition to the overall logic, we particulary leverage their *workflow* paradigm which is integral in the use of the napari plugin interface. Implementation of `infer-subc` as a Napari plugin using this framework is called [`organelle-segmenter-plugin`](https://github.com/ndcn/organelle-segmenter-plugin).

​The `Allen Cell & Structure Segmenter` is a Python-based open source toolkit developed at the Allen Institute for Cell Science for 3D segmentation of intracellular structures in fluorescence microscope images: [`aicssegmentation`](https://github.com/AllenCell/aics-segmentation) package. This toolkit brings together classic image segmentation and iterative deep learning workflows first to generate initial high-quality 3D intracellular structure segmentations and then to easily curate these results to generate the ground truths for building robust and accurate deep learning models. The toolkit takes advantage of the high replicate 3D live cell image data collected at the Allen Institute for Cell Science of over 30 endogenous fluorescently tagged human induced pluripotent stem cell (hiPSC) lines. Each cell line represents a different intracellular structure with one or more distinct localization patterns within undifferentiated hiPS cells and hiPSC-derived cardiomyocytes. Here, we leveraged select segmentation methods specialized for a particular organelle shape (i.e., round versus tubular objects) to carried out segmentation states within the [`infer-subc` segmentation workflows](/docs/segmentation.md).


## Issues
If you encounter any problems, please file an issue with a detailed description.

## Development
Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License
Distributed under the terms of the [BSD-3] license.

`infer-subc` and `organelle-segmenter-plugin` are free and open source software.

## Support of this project includes:
- [CZI Neurodegeneration Challenge Network (NDCN)](https://chanzuckerberg.com/science/programs-resources/neurodegeneration-challenge/)

# Publications

`infer-subc` analysis has been featured in:
1. Shannon N. Rhoads, Weizhen Dong, Chih-Hsuan Hsu, Ngudiankama R. Mfulama, Joey V. Ragusa, Michael Ye, Andy Henrie, Maria Clara Zanellati, Graham H. Diering, Todd J. Cohen, Sarah Cohen. *Neurons and astrocytes have distinct organelle signatures and responses to stress.* bioRxiv 2024.10.30.621066; doi: https://doi.org/10.1101/2024.10.30.621066
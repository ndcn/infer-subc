# Segmentation Workflows 🖍️
The starting point for the `infer-subc` analysis pipeline is to perform instance segmentation on single or multichannel confocal microscopy images, where each channel labels a different intracellular component. Each channel or organelle will be segmented from a single intensity channel. The subsequent analysis is performed at a single-cell level, so a mask of the cell area will also be carried out.

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


### <ins>Option B:</ins> [Notebooks](/docs/nbs/overview.md) 📚
The primary purpose of the Juptyer notebooks are to walk step-by-step through each of the segementation workflows. We hope these notebooks provide a more easily accessible resource for those who are new to or well versed in Python image analysis. *The notesbooks below include sets to segment a single image. A batch processing workflow has not yet been created.*

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
After batch processing, we recommend you quality check your segmentation results by visually inspecting the images. The Segmentation Validation pipeline is included in the [Full Quantification Pipeline Notebook](/notebooks/part_2_quantification/full_quantification_pipeline.ipynb) to streamline the validation process.

🚧 *In a future verions, this notebook will also include quality checks for assumptions made during quantification (i.e., only one nucleus and ER per cell, etc.).*
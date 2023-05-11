# reproducable imaging framework 🚧 WIP 🚧 

🚧 WIP 🚧 

SCohenLab 2D Image Processing  

Pipelines are [GET A GOOD DEFINITION / LINK].  They enable repoducability and scalabillity.

--------------
# PIPELINE FRAMEWORK OVERVIEW

As a general framework for our data processing pipeline we are defining 4 steps:
1. GOAL SETTING ✍
2. DATA CREATION
3. IMAGE PROCESSING  ⚙️🩻🔬
4. QUANTIFICATION 📏📐🧮

## 1. GOAL SETTING ✍

Here we make explicit what we are trying to accomplish.

## GOAL:  Infer sub-cellular components in order to understand interactome 
To measure shape, position, size, and interaction of eight organelles/cellular components (Nuclei (NU), Lysosomes (LS),Mitochondria (MT), Golgi (GL), Peroxisomes (PO), Endoplasmic Reticulum (ER), Lipid Droplet (LD), and CELLMASK) during differentiation of iPSCs, in order to understand the Interactome / Spatiotemporal coordination.

As part of goal setting we will also enumerate the specific tasks that need to be done to reach the goal.
#### summary of _OBJECTIVES_ ✅
- robust inference of subcellular objects:
  - 1️⃣-***cellmask***
  - 2️⃣-***nuclei***
  - 3️⃣-***cytoplasm***
  - 4️⃣-***lysosome***
  - 5️⃣-***mitochondria***
  - 6️⃣-***golgi***
  - 7️⃣-***peroxisome***
  - 8️⃣-***endoplasmic reticulum***
  - 9️⃣-***lipid body***

## 2. DATA CREATION
The second step is to get the data.  Capturing the data could be either from running the experiment, or mining a database.   Implicitly we need to also capture all the assumptions and methodologies, or _meta-data_.

> METHODS:📚📚
> 
> iPSC lines prepared and visualized on Zeiss Microscopes. 32 channel multispectral images collected.  Linear Unmixing in  ZEN Blue software with target emission spectra yields 8 channel image outputs.  Channels correspond to: Nuclei (NU), Lysosomes (LS),Mitochondria (MT), Golgi (GL), Peroxisomes (PO), Endoplasmic Reticulum (ER), Lipid Droplet (LD), and a “residual” signal.

> Meta-DATA 🏺 (artifacts)
>  - Microcope settings
>  - OME scheme
> - Experimenter observations
> - Sample, bio-replicate, image numbers, condition values, etc
>  - Dates
>  - File structure, naming conventions
>  - etc.

### DATA: linearly un-mixed flourescence images in .czi files

The bulk of the code in this repo is to solve this step:  processing the "raw" data to infer the locations of sub-cellular oranelles.

## 3. IMAGE PROCESSING  ⚙️🩻🔬
### INFERENCE OF SUB-CELLULAR OBJECTS
The imported images have already been pre-processed to transform the 32 channel spectral measuremnts into "linearly unmixed" images which estimate independently labeled sub-cellular components.  Thes 7 channels (plus a residual "non-linear" signal) will be used to infer the shapes and extents of these sub-cellular components.   
A single "optimal" Z slice is chosen for each image for subsequent 2D analysis.
We will perform computational image analysis on the pictures to _segment_ (or _binarize_) the components of interest for measurement.  In other procedures we can used these labels as "ground truth" labels to train machine learning models to automatically perform the inference of these objects.
Pseudo-independent processing of the imported multi-channel image to acheive each of the 9 objecives stated above.  i.e. infering: NUCLEI, CELLMASK, CYTOPLASM, LYSOSOME, MITOCHONDRIA, GOLGI COMPLEX, PEROZISOMES, ENDOPLASMIC RETICULUM, and LIPID BODIES

### General flow for infering objects via segmentation
- (extraction) 
- Pre-processing 🌒
- Core-processing (thresholding) 🌕
- Post-processing  🌘
- (post-postprocessing) 

### ~~QC 🚧 WIP 🚧~~ DEPRICATED

Finally, once we have inferred the organelle objects, we need to quantify them. These statistics, and the relationships among them will constitute the "interactome".

## 4. QUANTIFICATION 📏📐🧮

SUBCELLULAR COMPONENT METRICS
- general
  -  extent 
  -  size
  -  position
-  contacts (cross-stats)
-  radial projection and depth stats 
  - radial distribution (in cytosol)
  - depth distribution
  - zernike moments
# ADDITIONAL CONSIDERATIONS

## NOTE: PIPELINE TOOL AND DESIGN CHOICES?
We want to leverage the Allen Cell & Structure Segmenter.  It has been wrapped as a [napari-plugin](https://www.napari-hub.org/plugins/napari-allencell-segmenter) but fore the workflow we are proving out here we will want to call the `aicssegmentation` [package](https://github.com/AllenCell/aics-segmentation) directly.

## ​The Allen Cell & Structure Segmenter 
​The Allen Cell & Structure Segmenter is a Python-based open source toolkit developed at the Allen Institute for Cell Science for 3D segmentation of intracellular structures in fluorescence microscope images. This toolkit brings together classic image segmentation and iterative deep learning workflows first to generate initial high-quality 3D intracellular structure segmentations and then to easily curate these results to generate the ground truths for building robust and accurate deep learning models. The toolkit takes advantage of the high replicate 3D live cell image data collected at the Allen Institute for Cell Science of over 30 endogenous fluorescently tagged human induced pluripotent stem cell (hiPSC) lines. Each cell line represents a different intracellular structure with one or more distinct localization patterns within undifferentiated hiPS cells and hiPSC-derived cardiomyocytes.

More details about Segmenter can be found at https://allencell.org/segmenter
In order to leverage the A

---------------------
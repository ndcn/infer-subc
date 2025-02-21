# **Sample Data Folder** 📂

Upon cloning the **infer-subc** repository and following the steps in [notebook 1.0_image_setup](/notebooks/part_1_segmentation_workflows/1.0_image_setup.ipynb), the Sample Data Folder will contain **two raw microscopy images** (.tiff) **workflow settings** and the **Sample Data Information Text**. With these documents, a user can successfully run the infer-subc notebooks, starting with notebooks [1.1a](/notebooks/part_1_segmentation_workflows/1.1a_infer_masks_from-composite_single_cell.ipynb) and [1.1b](/notebooks/part_1_segmentation_workflows/1.1b_infer_masks_from-composite_multiple-cells.ipynb), to produce quantification and segmentation. In order to run infer-subc using the sample data, begin by running notebooks [1.1a](/notebooks/part_1_segmentation_workflows/1.1a_infer_masks_from-composite_single_cell.ipynb) and [1.1b](/notebooks/part_1_segmentation_workflows/1.1b_infer_masks_from-composite_multiple-cells.ipynb) to create the masks segmentations. Then run notebooks 1.2-7 twice (once per cell type). This can be done by user input in notebooks 1.2-7 after the imports section, by setting the `cell type` variable equal to **"neuron"** or **"astrocyte"**. Following this, all the notebooks in part 2 (2.1-6 and the full quantification pipeline) can operate correctly. Please note that for the [full quantification pipeline](/notebooks/part_2_quantification/full_quantification_pipeline.ipynb) to run correctly, the raw images must be copied to the `sample_data/batch_example/raw` directory. [Notebook 2.5](/notebooks/part_2_quantification/2.5_combined_and_batch_processing.ipynb) does this automatically, but if you wish, you can copy both of the files into the directory manually. The original raw image files and segmentation settings are from experiments featured in the paper [Neurons and astrocytes have distinct organelle signatures and responses to stress](https://www.biorxiv.org/content/10.1101/2024.10.30.621066v1) authored by Shannon Rhoades et al. In this paper infer-subc output for neurons and astrocytes were analyzed to identify significant differences their responses to stress.

## Contents of Sample Data Folder 🗂️
###### This section will detail the expected contents of the sample data folder after the infer-subc notebooks are run comprehensively. When applicable, each file will be linked the original stored in bioimage archive. However, there is no need to download the files (other than the two initial raw images) as the notebooks in part 1 and part 2 will create them as intended. If you wish to download the original files, this can be done by simply clicking the links.

### batch_example
###### This folder contains all of the files related to batch processing, quantification and quality checking 

- #### edit_seg
    ###### This folder stores the segmentations edited in the **full_quantification_pipeline** notebook. Folder will be empty unless the user chooses to edit a segmentation.
- #### final_seg
    ###### This folder stores the finalized segmentations selected for batch quantification in the **full_quantification_pipeline**.
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-cell.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-cell.tiff) (💾 205.2 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-ER.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-ER.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-golgi.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-golgi.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-LD.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-LD.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-lyso.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-lyso.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-mito.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-mito.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-nuc.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-nuc.tiff) (💾 205.2 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-perox.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-perox.tiff) (💾 102.6 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-cell.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-cell.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-ER.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-ER.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-golgi.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-golgi.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-LD.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-LD.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-lyso.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-lyso.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-mito.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-mito.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-nuc.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-nuc.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-perox.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-perox.tiff) (💾 193.8 MB)

- #### quant
    ###### This folder stores the quantification output from notebooks **2.5**, **2.6** and **full_quantification_pipeline**. The sample data batches the quantification of the astrocyte and neuron together to display functionality. Links are in reference to the original datasets where instead the observations were batched by cell type. Please note that **VD-0505** and **C2-121** are the names of the experimental replicates of the example **astrocyte** and the **neuron** respecitively. Additionally, in future versions of infer-subc the term **contacts** will be replaced by **interactions**.

    
    - files ending in `_contacts.csv`
        - [C2-121 interactions](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_quantification/C2-121_interactions.csv) (💾 53.8 MB)
        - [VD-0505 interactions](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_quantification/VD-0505_interactions.csv) (💾 16.3 MB)
    - files ending in `_distributions.csv`
        - [C2-121 distributions](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_quantification/C2-121_distributions.csv) (💾 8.5 MB)
        - [VD-0505 distributions](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_quantification/VD-0505_distributions.csv) (💾 3.0 MB)
    - files ending in `_organelles.csv`
        - [C2-121 organelles](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_quantification/C2-121_organelles.csv) (💾 27.1 MB)
        - [VD-0505 organelles](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_quantification/VD-0505_organelles.csv) (💾 7.5 MB)
    - files ending in `_regions.csv`
        - [C2-121 regions](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_quantification/C2-121_regions.csv) (💾 78 KB)
        - [VD-0505 regions](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_quantification/VD-0505_regions.csv) (💾 32 KB)
    - files ending in `_per_contact_summarystats.csv`
        - [Astrocytes interaction summary](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/Astrocyte_interaction_summarystats.csv) (💾 615 KB)
        - [Neurons interaction summary](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/Neuron_interactions_summarystats.csv) (💾 969 KB)
    - files ending in `_distribution_summarystats.csv`
        - [Astrocytes distribution summary](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/Astrocyte_distribution_summarystats.csv) (💾 821 KB)
        - [Neurons distribution summary](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/Neuron_distribution_summarystats.csv) (💾 1.4 MB)
    - files ending in `_per_org_summarystats.csv`
        - [Astrocytes organelle summary](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/Astrocyte_organelle_summarystats.csv) (💾 743 KB)
        - [Neurons organelle summary](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/Neuron_organelle_summarystats.csv) (💾 1.2 MB)
    - files ending in `_per_region_summarystats.csv`
        - [Astrocytes region summary](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/Astrocyte_region_summarystats.csv) (💾 35 KB)
        - [Neurons region summary](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/Neuron_region_summarystats.csv) (💾 66 KB)
    - files ending in `_summarystats_combined.csv`
        - [Neurons combined summary](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/Astrocyte_summarystats_1418-combined.csv) (💾 1.6 MB)
        - [Astrocytes combined summary](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/Neuron_summarystats_1418-combined.csv) (💾 2.5 MB)

- #### raw
    ###### This folder contains copies of the raw neuron and astrocyte images downloaded in the **1.0_image_setup** notebook.
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_deconvolution/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome.tiff) (💾 1.14 GB)
    - [20230727_C2-121_unconditioned_well 10_cell 4_25nM TG_Linear unmixing_0_cmle.ome.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_deconvolution/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome.tiff) (💾 2.16 GB)

- #### seg
    ###### This folder contains duplicates of the resulting segmentations created in the masks notebooks 1.1a & 1.1b as well as the segmentation notebooks 1.2-7.
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-ER.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-ER.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-golgi.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-golgi.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-LD.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-LD.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-lyso.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-lyso.tiff) (💾 102.6 MB)
    - 05052022_astro_control_2_Linear unmixing_0_cmle.ome-masks.tiff (originally seperated into nucleus and cellmask)
        - [nucleus](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-nuc.tiff) (💾 205.2 MB)
        - [cellmask](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-cell.tiff) (💾 205.2 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-mito.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-mito.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-perox.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-perox.tiff) (💾 102.6 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-ER.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-ER.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-golgi.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-golgi.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-LD.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-LD.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-lyso.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-lyso.tiff) (💾 193.8 MB)
    - 20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-masks.tiff (originally seperated into nucleus and cellmask)
        - [nucleus](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-nuc.tiff) (💾 193.8 MB)
        - [cellmask](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-cell.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-mito.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-mito.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-perox.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-perox.tiff) (💾 193.8 MB)

### example_astrocyte
###### This folder contains all of the files related to the segmentation of the example astrocyte

- #### raw
    ###### The intended location of the downloaded astrocyte from bioimage archive
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_deconvolution/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome.tiff) (💾 1.14 GB)
- #### seg
    ###### This folder contains the resulting segmentations created in notebook 1.1b as well as the segmentation notebooks 1.2-7.
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-ER.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-ER.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-golgi.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-golgi.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-LD.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-LD.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-lyso.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-lyso.tiff) (💾 102.6 MB)
    - 05052022_astro_control_2_Linear unmixing_0_cmle.ome-masks_B.tiff (originally seperated into nucleus and cellmask)
        - [nucleus](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-nuc.tiff) (💾 205.2 MB)
        - [cellmask](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-cell.tiff) (💾 205.2 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-mito.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-mito.tiff) (💾 102.6 MB)
    - [05052022_astro_control_2_Linear unmixing_0_cmle.ome-perox.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20astrocyte%20data/Z-stacks/replicate2_VD-0505/VD-0505_segmentations/05052022_astro_control_2_Linear%20unmixing_0_cmle.ome-perox.tiff) (💾 102.6 MB)
    

- #### settings
    ###### The settings (in JSON file type) used to create each of the corresponding segmentations in the seg folder. These settings can be used in napari to automatically apply the workflow settings.
    
    - ER.json
    - golgi.json
    - LD.json
    - lyso.json
    - masks_B.json
    - mito.json
    - perox.json

### example_neuron
###### This folder contains all of the files related to the segmentation of the example neuron

- #### raw
    ###### The intended location of the downloaded neuron from bioimage archive
    - [20230727_C2-121_unconditioned_well 10_cell 4_25nM TG_Linear unmixing_0_cmle.ome.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_deconvolution/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome.tiff) (💾 2.16 GB)
- #### seg
    ###### This folder contains the resulting segmentations created in notebook 1.1a as well as the segmentation notebooks 1.2-7.
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-ER.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-ER.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-golgi.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-golgi.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-LD.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-LD.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-lyso.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-lyso.tiff) (💾 193.8 MB)
    - 20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-masks_A.tiff (originally seperated into nucleus and cellmask)
        - [nucleus](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-nuc.tiff) (💾 193.8 MB)
        - [cellmask](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-cell.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-mito.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-mito.tiff) (💾 193.8 MB)
    - [20230727_C2-121_conditioned_well 4_cell 3_untreated_Linear unmixing_0_cmle.ome-perox.tiff](https://www.ebi.ac.uk/biostudies/files/S-BIAD1445/Neuron%20and%20astrocyte%20organelle%20signatures%20dataset/Primary%20rat%20neuron%20data/Z-stacks/replicate3_C2-121/C2-121_segmentations/20230727_C2-121_conditioned_well%204_cell%203_untreated_Linear%20unmixing_0_cmle.ome-perox.tiff) (💾 193.8 MB)
- #### settings
    ###### The settings (in JSON file type) used to create each of the corresponding segmentations in the seg folder. These settings can be used in napari to automatically apply the workflow settings.

    - ER.json
    - golgi.json
    - LD.json
    - lyso.json
    - masks_A.json
    - mito.json
    - perox.json




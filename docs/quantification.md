# Quantification Pipelines 🧮📐

After each of the organelles of interest are segmented, single or multi-organelle analysis can be carried out using Jupyter Notebook-based pipeline(s). Each of the following analysis types are modular and can be used in combination or separately.

### **<ins>Option A:</ins> Combined Analysis** 

The [full_quantification_pipeline](/notebooks/part_2_quantification/full_quantification_pipeline.ipynb) includes quantification of the `morphology`, `interactions`, and `distribution` of any number of organelles within the same cell. This is the current recommended form of the analysis and is demonstrated in the [full_quantification_pipeline]() Jupyter notebook.

### **<ins>Option B:</ins> Individual analysis pipelines (single cell analysis only)**

The following notebooks include steps to quantify features of single organelles or pairs of organelles (interactions) from individual cells. However, these notebooks primarily act as a step-by-step guide to understanding each measurement type. 
- [Organelle morphology](/notebooks/part_2_quantification/2.1_organelle_morphology.ipynb)
- [Pairwise organelle interactions](/notebooks/part_2_quantification/2.2_organelle_interactions.ipynb)
- [Subcellular distribution](/notebooks/part_2_quantification/2.3_distribution_measurements.ipynb)
- [Cell/nucleus morphology](/notebooks/part_2_quantification/2.4_cell_region_morphology.ipynb)
- COMBINED ANAYSIS ONLY: [batch processing](/notebooks/part_2_quantification/2.5_combined_and_batch_processing.ipynb)
- COMBINED ANAYSIS ONLY: [per-cell summary](/notebooks/part_2_quantification/2.6_summary_stats.ipynb)

🚧 *Future implimentations of the individual notebooks will include batch processing capabilities (e.g., multiple cells, multiple organelles).*

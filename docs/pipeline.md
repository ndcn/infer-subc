# general worfklow / data pipeline 

🚧 WIP 🚧 

# PUTATIVE WORKFLOW


## WORKFLOW EDITOR PLUGIN
- FINE-TUNE SEGMENTATIONS
  - export workflow.jsons
    - masks:
      - nuclei
      - cellmask
      - cytoplasm
    - organelles:
      - lyso
      - mito
      - golgi
      - perox
      - ER
      - LD


## BATCHPROCESS WORKFLOW
- BATCH PROCESS
  - load workflow.jsons for: 
  1. masks
    - export: masks .tiff as stack (nuclei, cellmask, cytoplasm)
  2. organelles
    - export individual .tiffs



## NOTEBOOK ~~OR ***FUTURE*** PLUGIN~~
- COLLECT ORGANELLE STATS
  - extract masks.tiffs as individual
    - nuclei, cellmask, cytoplasm
  - collect regionprops for all organelles
    - export .csvs

---------------------
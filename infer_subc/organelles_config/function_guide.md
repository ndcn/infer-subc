# Overview

## `conf_.json`s and `batch_.json`s


*** There are two _levels_ of `.json`s.   Ones intended to use step-by-step to choose parameters via the Workflow Editor Widget, and those intended to be run as batches in the Batch Processing Widget***  

### Workflow editing:  `conf_.json`s
Here the specific parameters of the organelle inference procedures can be fine-tuned and experimented with.  Inputs to most require first selecting a Z-slice, e.g. via `conf_get_zslice.json`.  

#### `conf_get_zslice.json`

#### `conf_infer_.json`s
Each organelle has its own procedure.
`conf_1.2.nuclei_stepbystep_from_raw.json`
`conf_1.3.cytoplasm_from_raw.json`
`conf_1.4.lysosome_stepbystep_from_raw.json`
`conf_1.5.mitochondria_stepbystep_from_raw.json`
`conf_1.6.golgi_stepbystep_from_raw.json`
`conf_1.7.peroxisome_stepbystep_from_raw.json`
`conf_1.8.er_stepbystep_from_raw.json`
`conf_1.9.lipid_stepbystep_from_raw.json`
`conf_2.1.soma_stepbystep.json`
`conf_2.2.nuclei_stepbystep.json`

### Batch editing:  `batch_.json`s
These are intended to import a full multi-channel, multi-zslice images, infer all of the organelles, and export the inferred objects. (And eventually the summary statistics from the inferred objects).  The `batch_.json` are composed of the sequence of `infer_organelle` functions which have fixed parameters.  These need to be edited by hand.  
TODO: make a widget to export the parameters from a `conf_.json` into a `batch_.json`
`conf_infer_fixed_infer_organelles_batch.json`
`conf_infer_fixed_infer_organelles_batch2.json`


#### export.
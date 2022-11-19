# Overview

## `conf_.json`s and `batch_.json`s


*** There are two _levels_ of `.json`s.   Ones intended to use step-by-step to choose parameters via the Workflow Editor Widget, and those intended to be run as batches in the Batch Processing Widget***  

### Workflow editing:  `conf_.json`s
Here the specific parameters of the organelle inference procedures can be fine-tuned and experimented with.  Inputs to most require first selecting a Z-slice, e.g. via `conf_get_zslice.json`.  

#### `conf_get_zslice.json`

#### `conf_infer_.json`s
Each organelle has its own procedure.


### Batch editing:  `batch_.json`s
These are intended to import a full multi-channel, multi-zslice images, infer all of the organelles, and export the inferred objects. (And eventually the summary statistics from the inferred objects).  The `batch_.json` are composed of the sequence of `infer_organelle` functions which have fixed parameters.  These need to be edited by hand.  
TODO: make a widget to export the parameters from a `conf_.json` into a `batch_.json`


#### export.
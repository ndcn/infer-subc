# organelles sub-module

The code in this sub-module have the routines to perform a (hopefully) robust inference of subcellular objects:

+ 1️⃣-***cellmask***
+ 2️⃣-***nuclei***
+ 3️⃣-***cytoplasm***
+ 4️⃣-***lysosome***
+ 5️⃣-***mitochondria***
+ 6️⃣-***golgi***
+ 7️⃣-***peroxisome***
+ 8️⃣-***endoplasmic reticulum***
+ 9️⃣-***lipid body***

From the results of the  1️⃣-***cellmask***, 2️⃣-***nuclei***, the  3️⃣-***cytoplasm***, a mask of the cytoplasm for each cell of interest is derived.

By "inference of sub-cellular objects" we mean assigning each pixel to belonging to an organell as estimated from the florescence image in the apropriate channel.  This is done here by image processing and thresholding.


## `organelles` submodule

::: infer_subc.organelles
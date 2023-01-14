# organelles

# organelles sub-module

The code in this sub-module have the routines to perform a (hopefully) robust inference of subcellular objects:
* 1️⃣-***soma***
- 2️⃣-***nuclei*** 
- 3️⃣-***cytosol***
- 4️⃣-***lysosome***
  -  5️⃣-***mitochondria***
  -  6️⃣-***golgi*** 
  -  7️⃣. ***peroxisome***
  -  8️⃣-***endoplasmic reticulum***
  -  9️⃣-***lipid body***

From the results of the  1️⃣-***soma***, 2️⃣-***nuclei***, the  3️⃣-***cytosol***, a mask of the cytosol for each cell of interest is derived.   

By "inference of sub-cellular objects" we mean assigning each pixel to belonging to an organell as estimated from the florescence image in the apropriate channel.  This is done here by image processing and thresholding.

::: infer_subc_2d.organelles
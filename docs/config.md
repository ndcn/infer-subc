# setup and configuration

We recommending using conda to create an environment with the apropriate dependencies. Note that `centrosome` is installed from a github [fork](https://github.com/ergonyc/centrosome) of the CellProfiler [repo](https://github.com/CellProfiler/centrosome) to avoid some gnarly dependency issues.  There is an [active pull request]( https://github.com/CellProfiler/centrosome/pull/115) which will hopefully fix this requirement soon.  

```bash
conda create -n napari10 python=3.10
conda activate napari10
conda install -c conda-forge ipython ipykernel pip notebook napari scipy scikit-learn matplotlib

pip install aicsimageio tifffile aicspylibczi aicssegmentation napari-aicsimageio
pip install git+https://github.com/ergonyc/centrosome.git


```
Finally install `infer_subc` and `organelle-segmenter-plugin`:

```bash
pip install infer_subc
pip install organelle_segmenter_plugin
```

Or if you prefer, clone the repos and make an editable local installation:

```bash
git clone https://github.com/ndcn/infer-subc.git
cd infer-subc
pip install -e .
```

# Getting Started
`infer-subc` and the related `organelle-segmenter-plugin` for [Napari](https://napari.org/stable/) are available on `PyPI` via: 

```
pip install infer-subc
pip install organelle-segmenter-plugin
```

## Setup ⚙️
We recommending using [`conda`](https://www.anaconda.com/) to create an environment with the apropriate dependencies.

```bash
conda create -n infer-subc python=3.10
conda activate infer-subc

pip install napari[all]
pip install CellProfiler
pip install scipy scikit-learn matplotlib
pip install aicsimageio 
pip install aicspylibczi
pip install aicssegmentation 
pip install napari-aicsimageio 
pip install centrosome
pip install --upgrade tifffile
pip install vispy
pip install matlab
```

Finally install `infer_subc` and `organelle-segmenter-plugin`:

```bash
pip install infer-subc
pip install organelle-segmenter-plugin
```

Or if you prefer, clone the repos and make an editable local installation:

```bash
git clone https://github.com/ndcn/infer-subc.git
cd infer-subc
pip install -e .
```
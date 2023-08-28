
conda create -n napari10 python=3.10
conda activate napari10
conda install -c conda-forge ipython ipykernel pip notebook napari scipy scikit-learn matplotlib

pip install aicsimageio tifffile aicspylibczi aicssegmentation napari-aicsimageio
pip install git+https://github.com/ergonyc/centrosome.git


pip install centrosome
pip install infer_subc
pip install .

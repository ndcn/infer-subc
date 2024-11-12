
conda create -n infer-subc python=3.10
conda activate infer-subc

pip install napari[all]

pip install scipy scikit-learn matplotlib
pip install aicsimageio
pip install aicspylibczi
pip install aicssegmentation 
pip install napari-aicsimageio 
pip install vispy
pip install matlab
pip install centrosome

pip install infer-subc
pip install organelle-segmenter-plugin

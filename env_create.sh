
# bioformats_jar was screwing things up. with OME
conda create -n napari-env python=3.9 pip notebook 
conda activate napari-env
pip install 'napari[all]'
pip install scipy scikit-learn matplotlib #jupyter
pip install aicsimageio 
pip install aicspylibczi
pip install aicssegmentation #downgrades napari and scikitlearn

pip install napari-aicsimageio  

pip install -e <path to infer-subc>
pip install -e <path to organelle-segmenter-plugin>


##  Execute this if installing via pypi (not yet available)
pip install infer_subc
pip install napari-infer-subc  

# # pip install opencv-python
# # pip install opencv-contrib-python
# pip install opencv-python-headless  # seems to already be installed

###### TEST FOR CENTROSOME
conda create -n cento python=3.9 -c conda-forge pip notebook napari scipy scikit-learn 

conda activate cento
pip install centrosome aicsimageio tifffile aicspylibczi aicssegmentation napari-aicsimageio

# pip install 'napari[all]'
# pip install scipy scikit-learn matplotlib #jupyter
# python3 -m pip install centrosome "matplotlib >=3.1.3"
# conda install -c bioconda centrosome

pip install centrosome
pip install aicsimageio 
pip install aicspylibczi
pip install aicssegmentation #downgrades napari and scikitlearn
pip install napari-aicsimageio  



conda create -n napari10 python=3.10
conda activate napari10
conda install -c conda-forge ipython ipykernel pip notebook napari scipy scikit-learn matplotlib
conda install matplotlib

pip install aicsimageio tifffile aicspylibczi aicssegmentation napari-aicsimageio
pip install git+https://github.com/ergonyc/centrosome.git


# for windows substitute "python -m pip install" for "pip install"

# conda create -n win_cento python=3.10 
# conda activate win_cento

# conda install -c conda-forge pip notebook 
# python -m pip install git+https://github.com/ergonyc/centrosome.git


# python -m pip install napari scipy scikit-learn aicsimageio aicspylibczi aicssegmentation #downgrades napari and scikitlearn

# # pip install 'napari[all]'
# python -m pip install -e <path to infer-subc
# python -m pip install -e <path to organelle-segmenter-plugin>






###### TEST FOR CENTROSOME
conda create -n cento python=3.9 -c conda-forge pip notebook napari scipy scikit-learn 

conda activate cento

# pip install 'napari[all]'
# pip install scipy scikit-learn matplotlib #jupyter
# python3 -m pip install centrosome "matplotlib >=3.1.3"
# conda install -c bioconda centrosome

pip install centrosome
pip install aicsimageio 
pip install aicspylibczi
pip install aicssegmentation #downgrades napari and scikitlearn

pip install napari-aicsimageio  



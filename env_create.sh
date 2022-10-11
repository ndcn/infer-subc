
# bioformats_jar was screwing things up. with OME
conda create -n napari-env python=3.9 pip notebook 
conda activate napari-env
pip install 'napari[all]'
pip install scipy scikit-learn matplotlib #jupyter
pip install aicsimageio 
pip install aicspylibczi
pip install aicssegmentation #downgrades napari and scikitlearn

pip install napari-aicsimageio  

<<<<<<< HEAD
pip install infer_subc_2d
pip install napari-infer-subc  

# pip install opencv-python
# pip install opencv-contrib-python
pip install opencv-python-headless  # seems to already be installed

=======
>>>>>>> parent of 08c72e0 (fix 09_lipid body nb)

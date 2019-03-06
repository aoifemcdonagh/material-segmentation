# MINC_model_testing
Set of files for testing models pretrained on MINC dataset

**These scripts are still under development and subject to large changes.**

# classify_resized.py
Performs coarse classification at three scales as defined in MINC paper. Produces a probability map for each material class. Uses methods from full_image_classify.py

# full_image_classify.py
Takes a single image and performs coarse material classification
Arguments: 
1. `-i` or `--image` : path to image to perform material classfication on
2. `-m` or `--model` : directory path containing `.caffemodel` and `.prototxt` files
3. `-p` or `--plot` : set to `False` to avoid plotting results. Default is `True`

# Dependencies
python 2.7
caffe 1.0.0
numpy 1.14.3+
matplotlib 2.2.3+
scipy 0.17.0+
argparse 1.2.1+
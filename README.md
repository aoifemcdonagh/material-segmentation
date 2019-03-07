# MINC_model_testing
Set of files for testing models pretrained on MINC dataset

**These scripts are still under development and subject to large changes.**

# Dependencies
* python 2.7
* caffe 1.0.0

   Installation guide for Caffe: https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-Installation-Guide 


* numpy 1.14.3+
* matplotlib 2.2.3+
* scipy 0.17.0+
* argparse 1.2.1+

# Scripts

## classify_resized.py
Performs coarse classification at three scales as defined in MINC paper. Produces a probability map for each material class. Uses methods from full_image_classify.py

## full_image_classify.py
Takes a single image and performs coarse material classification
Arguments: 
1. `-i` or `--image` : path to image to perform material classfication on
2. `-m` or `--model` : directory path containing `.caffemodel` and `.prototxt` files
3. `-p` or `--plot` : set to `False` to avoid plotting results. Default is `True`

## fc_to_conv.py
Script for converting a GoogLeNet model to be fully convolutional (i.e. fully connected layers in the model are cast as convolutional layers). Assumes models are in the current working directory. Change model names/paths as necessary.

## test_fully_conv.py
Script for testing a fully conv network on image of arbitrary size. Change model paths as necessary. First script parameter is used to specify path to image to perform inference on.
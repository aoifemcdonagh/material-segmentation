# Material Segmentation using MINC dataset
This repo contains code written for a project as part of my Master's thesis. 
The project aimed to develop a method of estimating room acoustic properties from images. 


Image segmentation is implemented based on material classification. A GoogLeNet model pretrained on the Materials In Context Database is used to perform material classification. 
The material segmentation map is used to estimate sound absorption in a space based on absorption coefficients of identified materials.

The **material_segmentation** directory contains:
* **Image segmentation app**
* Script for converting a GoogLeNet model to be fully convolutional
* Modules for performing image segmentation based on material

The **ncs_demos** directory contains scripts for performing material segmentation demos using the **Intel Neural Compute Stick** (NCS). 
These scripts were developed for live demos on a handheld device. 
A demonstration device was constructed consisting of a Raspberry Pi, NCS, Pi camera, touchscreen and a custom printed case.

![alt text](https://github.com/aoifemcdonagh/material-segmentation/src/images/demo_setup.png)

## Dependencies

Dependency | Install Guide/Notes
-----------|--------------
python 3.5+ |
CUDA | https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
CuDNN | https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
caffe 1.0.0 | https://github.com/adeelz92/Install-Caffe-on-Ubuntu-16.04-Python-3 Follow steps carefully since they depend on your CUDA, CuDNN and python versions
OpenCV | Install using pip3, not during OpenVino install.
OpenVino | https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick Note untick OpenCV


## GUI
`run_gui.py` sets up continuous segmentation of a video file or camera input. Results are displayed in a TkInter GUI

`SegmentationApp.py` Contains SegmentationApp class which handles creation of GUI objects and threads for running image segmentation.

## Test Scripts

##### test_classification_simple.py
Script to classify material in a full image.
Uses MINC GoogLeNet model modified to be fully convolutional
Arguments:
   `--image` `-i` path to image to classify
   `--caffemodel` path to .caffemodel file
Example execution `python3 test_classification_simple.py -i image.jpg --caffemodel ../models/minc-googlenet-conv.caffemodel`

##### test_continuous_upsampling.py
This script performs continuous segmentation with pyramidal upsampling on video/camera stream
Arguments: 
   `--input` `-i` 'cam' or path to video file
   `--model` `-m` Path to an .xml file of a trained model (optional)
   `--padding` `-p` Number of pixels of padding to add (optional)
Example execution `python3 test_continuous_upsampling.py -i samplevid.mp4 -p 100`

##### test_fully_conv.py
Runs inference on a fully convolutional network with input image of arbitrary size. Image path is first argument. 
Example execution `python3 test_fully_conv.py image.jpg`

##### test_gpu_segmentation.py
Performs material segmentation on an image. Image path is first argument.
A "pyramidal" classification and upsampling technique is used. Input image is resized at three scales (as defined in MINC paper), inference is performed on resized images, outputs are upsampled and averaged. This technique produces a much smoother, higher resolution segmentation map. This script plots the segmentation results and a confidence map.
Example execution `python3 test_gpu_segmentation.py image.jpg`

##### test_padding.py
Verify the add_padding() and remove_padding() methods work correctly. Adds padding to an input image and plots it.
Example execution `python3 test_padding.py image.jpg 50`

##### test_upsampling.py
Performs segmentation on a single image using th pyramidal classification & upsampling approach. Arguments are the input image path and number of pixels of padding.
Example execution `python3 test_upsampling.py image.jpg 50`


## Demo Scripts
The `ncs_demos` directory contains scripts to run demos using the Movidius Neural Compute Stick (NCS). Also in this directory are python files containing functions required by multiple demo scripts, e.g. `ncs_utilities.py`. The structure of these modules is not final, and subject to futher development.

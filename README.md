# Material Segmentation using MINC dataset
This repo contains code written for a project as part of my Master's thesis. 
The project aimed to develop a method of estimating room acoustic properties from images. 

## Overview

Image segmentation is implemented based on material classification. 
A GoogLeNet model pretrained on the Materials In Context Database is used to perform material classification. 
The resulting material segmentation map is used to estimate sound absorption in a space based on absorption coefficients of identified materials.

The **material_segmentation** directory contains:
* **Image segmentation app**
* Script for converting a GoogLeNet model to be fully convolutional
* Modules for performing image segmentation based on material

The **ncs_demos** directory contains scripts for performing material segmentation demos using the **Intel Neural Compute Stick** (NCS). 
These scripts were developed for live demos on a handheld device. 
A demonstration device was constructed consisting of a Raspberry Pi, NCS, Pi camera, touchscreen and a custom printed case.

![alt text](https://github.com/aoifemcdonagh/material-segmentation/blob/master/src/pictures/demo_setup.png "demo setup")

## Dependencies

Dependency | Install Guide/Notes
-----------|--------------
python 3.5+ |
CUDA | https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
CuDNN | https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
caffe 1.0.0 | https://github.com/adeelz92/Install-Caffe-on-Ubuntu-16.04-Python-3 Follow steps carefully since they depend on your CUDA, CuDNN and python versions
OpenCV | Install using pip3, not during OpenVino install.
OpenVino | https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick Note untick OpenCV


## Image Segmentation App
An simple app was built to perform live image segmentation of frames from a video or camera feed.
The input stream (video or camera) plays until a user wants a (material) class map to be generated.
An absorption coefficient map can also be generated.

![alt text](https://github.com/aoifemcdonagh/material-segmentation/blob/master/src/pictures/live_image_workflow.png "start up")

### runGUI_GPU.py  
This script runs the Segmentation App. 
A video file or camera input is displayed until a user specifies when to perform segmentation. 
Results are displayed by a TkInter GUI  
Arguments:  
   `-m` `--model` Path to a .caffemodel file. If no model path specified, a model path in `gpu_segment.py` where segmentation occurs.  
   `-i` `--input` 'cam' or path to an image  
   `-p` `--padding` number of pixels of padding to add. Default is 0 
   
Example execution:  
`python3 runGUI_GPU.py -i cam -p 200`

**GUI Start**   
![alt text](https://github.com/aoifemcdonagh/material-segmentation/blob/master/src/pictures/start.png "start up")

**Original frame**
![alt text](https://github.com/aoifemcdonagh/material-segmentation/blob/master/src/pictures/demo_table.png "demo original frame")  

**Material Segmentation**  
![alt text](https://github.com/aoifemcdonagh/material-segmentation/blob/master/src/pictures/demo_table_material.png "demo material segmentation")  

**Sound Absorption 'Heatmap'**  
![alt text](https://github.com/aoifemcdonagh/material-segmentation/blob/master/src/pictures/demo_table_abs.png "demo sound absorption heatmaps")  

### SegmentationApp.py  
Contains SegmentationApp class which handles creation of GUI objects and threads for running image segmentation.

## Test Scripts 
The `tests` directory contains scripts for testing various functions within this project.

## Demo Scripts
The `ncs_demos` directory contains scripts to run demos using the Movidius Neural Compute Stick (NCS). Also in this directory are python files containing functions required by multiple demo scripts, e.g. `ncs_utilities.py`. The structure of these modules is not final, and subject to futher development.

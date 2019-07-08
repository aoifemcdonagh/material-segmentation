# Test Scripts
This directory contains scripts for testing various functions of this project.

##### test_classification_simple.py
Script to classify material in a full image.
Uses MINC GoogLeNet model modified to be fully convolutional  
Arguments:  
   `--image` `-i` path to image to classify  
   `--caffemodel` path to .caffemodel file   
Example execution  
`python3 test_classification_simple.py -i image.jpg --caffemodel ../models/minc-googlenet-conv.caffemodel`

##### test_continuous_upsampling.py
This script performs continuous segmentation with pyramidal upsampling on video/camera stream  
Arguments:  
   `--input` `-i` 'cam' or path to video file  
   `--model` `-m` Path to an .xml file of a trained model (optional)  
   `--padding` `-p` Number of pixels of padding to add (optional)    
Example execution  
`python3 test_continuous_upsampling.py -i samplevid.mp4 -p 100`

##### test_fully_conv.py
Runs inference on a fully convolutional network with input image of arbitrary size. Image path is first argument.  
Example execution  
`python3 test_fully_conv.py image.jpg`

##### test_gpu_segmentation.py
Performs material segmentation on an image. Image path is first argument.
A "pyramidal" classification and upsampling technique is used. Input image is resized at three scales (as defined in MINC paper), inference is performed on resized images, outputs are upsampled and averaged. This technique produces a much smoother, higher resolution segmentation map. This script plots the segmentation results and a confidence map.  
Example execution  
`python3 test_gpu_segmentation.py image.jpg`

##### test_padding.py
Verify the add_padding() and remove_padding() methods work correctly. Adds padding to an input image and plots it.  
Example execution  
`python3 test_padding.py image.jpg 50`

##### test_upsampling.py
Performs segmentation on a single image using th pyramidal classification & upsampling approach. Arguments are the input image path and number of pixels of padding.  
Example execution  
`python3 test_upsampling.py image.jpg 50`

##### test_abs_coeff.py
UNFINISHED
Test the calculation of average absorption coefficient from segmentation results.
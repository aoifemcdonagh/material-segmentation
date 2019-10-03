"""
Script to classify material in a full image.
Uses MINC GoogLeNet model modified to be fully convolutional
Inputs:
     - path to image to classify
Optional inputs:
     - path to .caffemodel file
"""

import argparse
from material_segmentation import minc_plotting
import caffe
from material_segmentation.gpu_segment import classify

caffe.set_mode_gpu()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, help="path to image to be classified")
parser.add_argument("--caffemodel", type=str, default="models/minc-googlenet-conv.caffemodel", help="path to caffemodel file")

args = parser.parse_args()

image = caffe.io.load_image(args.image)  # must load image using caffe.io.load_image(). note this outputs RGB image
output = classify(image, caffemodel=args.caffemodel)

minc_plotting.plot_class_map(output)
minc_plotting.plot_confidence_map(output)

"""
Script to classify material in a full image.
Uses MINC GoogLeNet model modified to be fully convolutional
Inputs:
     - path to image to classify
Optional inputs:
     - path to .prototxt file
     - path to .caffemodel file
     - option to plot results or not
"""

import argparse
import minc_plotting as minc_plot
import caffe
from gpu_segment import classify

caffe.set_mode_gpu()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, help="path to image to be classified")
parser.add_argument("--prototxt", type=str, default="models/deploy-googlenet-conv.prototxt", help="path to prototxt file")
parser.add_argument("--caffemodel", type=str, default="models/minc-googlenet-conv.caffemodel", help="path to caffemodel file")

args = parser.parse_args()

image = caffe.io.load_image(args.image)  # must load image using caffe.io.load_image(). note this outputs RGB image
output = classify(image, args.prototxt, args.caffemodel)

minc_plot.plot_class_map(output)
minc_plot.plot_confidence_map(output)
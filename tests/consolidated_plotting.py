"""
Script for generating plots for thesis diagrams
"""

from material_segmentation import minc_plotting, gpu_segment
import argparse
import caffe

caffe.set_mode_gpu()

parser = argparse.ArgumentParser()
parser.add_argument("mode", type=str, choices=["segment", "classify"], default="segment")
parser.add_argument("-i", "--image", type=str, help="path to image to be classified")
parser.add_argument("--caffemodel", type=str, default="models/minc-googlenet-conv.caffemodel", help="path to caffemodel file")
parser.add_argument("--padding", "-p", type=int, default=0, help="number of pixels to pad image with before segmenting")
parser.add_argument("--path", type=str, default="plots")
args = parser.parse_args()

image = caffe.io.load_image(args.image)  # must load image using caffe.io.load_image(). note this outputs RGB image

if args.mode == "segment":
    output = gpu_segment.segment(image, caffemodel=args.caffemodel, pad=args.padding)

else:  # Classify
    output = gpu_segment.classify(image, caffemodel=args.caffemodel)

minc_plotting.plot_output(output)



"""
Script for generating plots for thesis diagrams
"""

def get_aspect(shape):
    """
    Generate suitable plot resolution based on image dimensions
    :param shape: image dimensions
    :return: aspect ratio for plots
    """

    ratio = shape[1]/shape[0]  # W:H ratio for image
    #aspect = (ratio:ratio*)

   #return aspect



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
#aspect = get_aspect(image.shape)

if args.mode == "segment":
    output = gpu_segment.segment(image, caffemodel=args.caffemodel, pad=args.padding)

else:  # Classify
    output = gpu_segment.classify(image, caffemodel=args.caffemodel)

minc_plotting.plot_output(output, image=image, aspect=(25, 5))



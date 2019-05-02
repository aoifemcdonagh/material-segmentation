# Script to test GPU segmentation via pyramidal method.
import sys
import numpy as np
import minc_plotting as minc_plot
from gpu_segment import segment
import skimage

image_path = sys.argv[1]  # path to image to be segmented

# Equivalent to caffe.io.load_image(image_path)
orig_image = skimage.img_as_float(skimage.io.imread(image_path, as_grey=False)).astype(np.float32)
padding = 0
results = segment(orig_image, pad=padding)

# minc_plot.plot_probability_maps(av_prob_maps, results)
minc_plot.plot_class_map(results)
minc_plot.plot_confidence_map(results)
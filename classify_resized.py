#  Script for classifying an image at three scales as defined in the MINC paper
#  Uses the classify method in full_image_classify.py

import caffe
import matplotlib.pyplot as plt
import sys
import os
#import full_image_classify
from scipy import misc
import numpy as np


"""
    TODO: Function needs to be implemented which performs whole image segmentation on an image at three different
    scales. What should be the output based on next stage of processing? Probability maps for a whole image? seperate
    probability maps for each class for each image?
    possibly give method a name which better reflects its function
"""


#def classify():


"""

"""


def resize_image(im_path):
    scale = [1.0/np.sqrt(2), 1.0, np.sqrt(2)]
    # load input and configure preprocessing
    im = misc.imread(im_path)
    im1 = misc.imresize(im, size=scale[0], interp='bilinear')
    im2 = misc.imresize(im, size=scale[1], interp='bilinear')
    im3 = misc.imresize(im, size=scale[2], interp='bilinear')

    dir, file_name = os.path.split(im_path)
    im_name, ext = os.path.splitext(file_name)

    misc.imsave(os.path.join(dir, (im_name + "_1" + ext)), im1)
    misc.imsave(os.path.join(dir, (im_name + "_2" + ext)), im2)
    misc.imsave(os.path.join(dir, (im_name + "_3" + ext)), im3)


if __name__ == "__main__":
    caffe.set_mode_gpu()
    path = sys.argv[1]  # path to image to be segmented
    resize_image(path)
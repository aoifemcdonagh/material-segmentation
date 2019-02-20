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
    TODO: Function needs to be implemented which performs material classification on an image at three different
    scales. What should be the output based on next stage of processing? Probability maps for a whole image? Separate
    probability maps for each class for each image?
    possibly give method another name which better reflects its function

    This method should call 'classify()' function from full_image_classify
"""


#def classify():


"""
    Function for resizing and saving an image
    TODO: decide on paths to save images to wrt function of all other scripts
"""


def resize_image(im_path):
    scale = [1.0/np.sqrt(2), 1.0, np.sqrt(2)]  # Define scales as per MINC paper
    im = misc.imread(im_path)  # load image
    im1 = misc.imresize(im, size=scale[0], interp='bilinear')
    im2 = misc.imresize(im, size=scale[1], interp='bilinear')
    im3 = misc.imresize(im, size=scale[2], interp='bilinear')

    dir, file_name = os.path.split(im_path)  # Get directory path and full file name of original image
    im_name, ext = os.path.splitext(file_name)  # Get file name and extension of original image

    # Use path info from original image to save resized images
    misc.imsave(os.path.join(dir, (im_name + "_1" + ext)), im1)
    misc.imsave(os.path.join(dir, (im_name + "_2" + ext)), im2)
    misc.imsave(os.path.join(dir, (im_name + "_3" + ext)), im3)


if __name__ == "__main__":
    caffe.set_mode_gpu()
    path = sys.argv[1]  # path to image to be segmented
    resize_image(path)

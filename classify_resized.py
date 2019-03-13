#  Script for classifying an image at three scales as defined in the MINC paper
#  Uses the classify method in full_image_classify.py
#  Creates directory for resized images and results

import caffe
import sys
import os
import skimage

import full_image_classify as minc_utils
from scipy import misc
import numpy as np
from datetime import datetime

SCALES = [1.0 / np.sqrt(2), 1.0, np.sqrt(2)]  # Define scales as per MINC paper


def segment(im_path, results=None):
    """
    TODO: Function needs to be implemented which performs material classification on an image at three different
    scales. What should be the output based on next stage of processing? Probability maps for a whole image? Separate
    probability maps for each class for each image?
    possibly give method another name which better reflects its function

    This function currently does too much??

    This method should call 'classify()' function from full_image_classify

    :param im_path: path to image to segment
    :param results: directory path to store results
    :return:
    """

    orig_image = misc.imread(im_path)  # load image

    resized_images = get_resized_images(orig_image)  # Resize original images

    outputs = [minc_utils.classify(image) for image in resized_images]  # Perform classification on images

    av_prob_maps = prepare_prob_maps(orig_image, outputs)

    minc_utils.plot_probability_maps(av_prob_maps, results)


def prepare_prob_maps(im, outputs):
    """
    :param im: original image (needed for shape)
    :param outputs: List of outputs from
    :return: Probability maps for each class, averaged from resized images probability maps
    """

    # Get probability maps for each class for each image
    prob_maps = [minc_utils.get_probability_maps(out) for out in outputs]

    # Upsampling probability maps to be same dimensions as original image
    # np.array so that they can be averaged later
    upsampled_prob_maps = np.array([[skimage.transform.resize(prob_map,
                                                              output_shape=(im.shape[0], im.shape[1]),
                                                              mode='constant',
                                                              cval=0,
                                                              preserve_range=True)
                                     for prob_map in prob_maps_single_image]
                                    for prob_maps_single_image in prob_maps])

    # Probability maps for each class, averaged from resized images probability maps
    averaged_prob_maps = np.average(upsampled_prob_maps, axis=0)

    return averaged_prob_maps


def get_resized_images(im):
    """
    Function for resizing an image to scales as defined in MINC paper
    Not saving images anymore, just returning them for processing

    :param im: pre-loaded image
    :return: resized images
    """

    # Return the resized images

    return [skimage.transform.rescale(im, scale, mode='constant', cval=0) for scale in SCALES]


if __name__ == "__main__":
    caffe.set_mode_gpu()
    image_path = sys.argv[1]  # path to image to be segmented
    segment(image_path)

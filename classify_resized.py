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

    This method should call 'classify()' function from full_image_classify

    Inputs:
        - im_path: path to image to segment
        - results: directory path to store results
    """

    im = misc.imread(im_path)  # load image
    im_files = resize_image(im_path, results)  # perform image resizing

    outputs = [minc_utils.classify(image) for image in im_files]  # Perform classification on images
    prob_maps = [minc_utils.get_probability_maps(out) for out in outputs]  # Get probability maps for each class for each image

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

    minc_utils.plot_probability_maps(averaged_prob_maps, results)

    # Upscale output to have fixed smaller dimension of 550


def resize_image(im_path, results=None):
    """
    Function for resizing and saving an image

    :return: paths to resized images

    TODO: decide on paths to save images to wrt function of all other scripts
    """

    if results is None:  # create directory path for test results if none specified
        results = os.path.join(os.getcwd(), 'results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    os.makedirs(results)  # Create the results directory

    im = misc.imread(im_path)  # load image
    _, file_name = os.path.split(im_path)  # Get directory path and full file name of original image
    im_name, ext = os.path.splitext(file_name)  # Get file name and extension of original image

    im_paths = []  # Empty list to store paths

    for scale in SCALES:  # Resize input image at specified scales
        # Note that resized images labelled with "0,1,2.." instead of the actual scale b/c scale is float
        im_paths.append(os.path.join(results, (im_name + str(SCALES.index(scale)) + ext)))  # Create path to save image
        resized_image = misc.imresize(im, size=scale, interp='bilinear')  # Resize image
        misc.imsave(im_paths[SCALES.index(scale)], resized_image)  # Save resized image

    # Return the paths to resized images
    return im_paths


if __name__ == "__main__":
    caffe.set_mode_gpu()
    image_path = sys.argv[1]  # path to image to be segmented
    segment(image_path)

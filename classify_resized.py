#  Script for classifying an image at three scales as defined in the MINC paper
#  Uses the classify method in full_image_classify.py
#  Creates directory for resized images and results

import caffe
import sys
import os
import full_image_classify as minc_utils
from scipy import misc
import numpy as np
from datetime import datetime

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


def segment(im_path, results):
    im = misc.imread(im_path)  # load image
    im_files = resize_image(im_path, results)  # perform image resizing

    outputs = [minc_utils.classify(image) for image in im_files]  # Perform classification on images
    all_prob_maps = [minc_utils.get_probability_maps(out) for out in outputs]  # Get probability maps for each class for each image

    # Upsample each output probability map (to original image size??)
    upsampled_prob_maps = np.array([[misc.imresize(prob_map, size=(im.shape[0], im.shape[1]), interp='bilinear') for prob_map in prob_maps] for prob_maps in all_prob_maps])
    averaged_prob_maps = np.average(upsampled_prob_maps, axis=0)  # Probability maps for each class, averaged from resized images probability maps

    minc_utils.plot_probability_maps(averaged_prob_maps, results)

    print("stop")
    # Upscale output to have fixed smaller dimension of 550



"""
    Function for resizing and saving an image
    TODO: decide on paths to save images to wrt function of all other scripts
"""


def resize_image(im_path, results):
    scales = {'1': 1.0/np.sqrt(2), '2': 1.0, '3': np.sqrt(2)}  # Define scales as per MINC paper
    im = misc.imread(im_path)  # load image
    _, file_name = os.path.split(im_path)  # Get directory path and full file name of original image
    im_name, ext = os.path.splitext(file_name)  # Get file name and extension of original image

    # Save new images in tests directory
    for num, scale in scales.iteritems():
        misc.imsave(os.path.join(results, (im_name + num + ext)), misc.imresize(im, size=scale, interp='bilinear'))

    return [os.path.join(results, (im_name + num + ext)) for num in scales.iterkeys()]  # return paths to new images

if __name__ == "__main__":
    caffe.set_mode_gpu()
    image_path = sys.argv[1]  # path to image to be segmented
    results_dir = os.path.join(os.getcwd(), 'results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))  # create directory for test results
    os.makedirs(results_dir)
    segment(image_path, results_dir)

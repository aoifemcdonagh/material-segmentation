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

SCALES = [1.0/np.sqrt(2), 1.0, np.sqrt(2)]  # Define scales as per MINC paper

def segment(im_path, results):
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

    # Upsample each output probability map (to original image size??)
    #upsampled_prob_maps = np.array([[misc.imresize(prob_map, size=(im.shape[0], im.shape[1]), interp='bilinear') for prob_map in prob_maps] for prob_maps in prob_maps])

    #upsampled_prob_maps_ski = np.array([[skimage.transform.rescale(prob_map, scale=(im.shape[0], im.shape[1]), mode='constant', cval=0) for prob_map in prob_maps] for prob_maps in images_prob_maps])
    upsampled_prob_maps_ski = np.empty_like(prob_maps)

    for prob_maps_single_scale in prob_maps:
        for prob_map in prob_maps_single_scale:
            i = prob_maps_single_scale.index(prob_map)  # Get index of current set of probability maps
            j = prob_maps.index(prob_maps_single_scale)  # Get index of individual prob map in current set of prob maps
            upsampled_prob_maps_ski[i][j] = skimage.transform.rescale(prob_map, scale=SCALES[i], mode='constant', cval=0)

    for i in range(0, prob_maps.shape[0]):
        image_prob_maps = prob_maps[i]
        for j in range(0, image_prob_maps.shape[0]):
            prob_map = image_prob_maps[j]
            upsampled_prob_maps_ski[i][j] = \
                skimage.transform.rescale(prob_map, scale=(im.shape[0], im.shape[1]), mode='constant', cval=0)

    averaged_prob_maps = np.average(upsampled_prob_maps_ski, axis=0)  # Probability maps for each class, averaged from resized images probability maps

    minc_utils.plot_probability_maps(averaged_prob_maps, results)

    print("stop")
    # Upscale output to have fixed smaller dimension of 550


def resize_image(im_path, results):
    """
        Function for resizing and saving an image
        TODO: decide on paths to save images to wrt function of all other scripts
    """

    im = misc.imread(im_path)  # load image
    _, file_name = os.path.split(im_path)  # Get directory path and full file name of original image
    im_name, ext = os.path.splitext(file_name)  # Get file name and extension of original image

    # Save new images in tests directory
    for scale in SCALES:
        misc.imsave(os.path.join(results, (im_name + str(SCALES.index(scale)) + ext)), misc.imresize(im, size=scale, interp='bilinear'))

    return [os.path.join(results, (im_name + str(num) + ext)) for num in range(0, len(SCALES))]  # return paths to new images

if __name__ == "__main__":
    caffe.set_mode_gpu()
    image_path = sys.argv[1]  # path to image to be segmented
    results_dir = os.path.join(os.getcwd(), 'results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))  # create directory for test results
    os.makedirs(results_dir)
    segment(image_path, results_dir)

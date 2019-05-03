#  Script for classifying an image at three scales as defined in the MINC paper
#  Uses the classify method in minc_classify.py
#  Creates directory for resized images and results

import skimage
import minc_classify as minc_utils
import numpy as np

SCALES = [1.0 / np.sqrt(2), 1.0, np.sqrt(2)]  # Define scales as per MINC paper


def segment(im, pad=0):
    """
    TODO: Function needs to be implemented which performs material classification on an image at three different
    scales. What should be the output based on next stage of processing? Probability maps for a whole image? Separate
    probability maps for each class for each image?
    possibly give method another name which better reflects its function

    This function currently does too much??

    This method should call 'classify()' function from full_image_classify

    :param im: image to segment
    :param pad: number of pixels of padding to add
    :return:
    """

    padded_image = add_padding(im, pad)  # Add padding to original image
    resized_images = resize_images(padded_image)  # Resize original images

    outputs = [minc_utils.classify(image) for image in resized_images]  # Perform classification on images

    average_prob_maps = get_average_prob_maps(outputs, im.shape, pad)

    return average_prob_maps


def get_average_prob_maps(network_outputs, shape, pad=0):
    """
    :param network_outputs: List of outputs
    :param shape: shape of original image needed for upsampling
    :return: Probability maps for each class, averaged from resized images probability maps
    """

    # Get probability maps for each class for each image
    prob_maps = [minc_utils.get_probability_maps(out) for out in network_outputs]

    # Upsample probability maps to dimensions of original image (plus any padding)
    upsampled_prob_maps = upsample(prob_maps, output_shape=(shape[0] + pad*2, shape[1] + pad*2))

    # Probability maps for each class, averaged from resized images probability maps
    averaged_prob_maps = np.average(upsampled_prob_maps, axis=0)

    # Remove the padded sections from the averaged prob maps
    averaged_prob_maps = [remove_padding(prob_map, pad) for prob_map in averaged_prob_maps]

    return np.array(averaged_prob_maps)


def upsample(prob_maps_multiple_images, output_shape):
    """
    Function for performing upsamping of probability maps
    :param prob_maps: Probability maps for each class for each resized image
    :param output_shape: Desired shape to upsample to (should be dimensions of original image)
    :return:
    """

    # Upsampling probability maps to be same dimensions as original image
    # np.array so that they can be averaged later
    return np.array([[skimage.transform.resize(prob_map,
                                               output_shape=output_shape,
                                               mode='constant',
                                               cval=0,
                                               preserve_range=True)
                      for prob_map in prob_maps_single_image]
                    for prob_maps_single_image in prob_maps_multiple_images])


def resize_images(im):
    """
    Function for resizing an image to scales as defined in MINC paper
    Not saving images anymore, just returning them for processing

    :param im: pre-loaded image
    :return: list of resized images
    """

    return [skimage.transform.rescale(im, scale, mode='constant', cval=0) for scale in SCALES]


def add_padding(im, pad=0):
    """
    Function for padding image before classification
    :param im: image (preloaded with caffe.io.load_image)
    :param pad: number of pixels of padding to add (default 0)
    :return: image with padding
    """

    return np.pad(im, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='symmetric')


def remove_padding(im, pad=0):
    """
    Function for removing padding from an image
    :param im: image or prob map to remove padding from
    :param pad: number of pixels of padding to remove
    :return:
    """

    if pad == 0:
        return im
    else:
        return im[pad:-pad, pad:-pad]


if __name__ == "__main__":
    import os
    import sys
    from datetime import datetime
    import minc_plotting as minc_plot

    image_path = sys.argv[1]  # path to image to be segmented

    # Equivalent to caffe.io.load_image(image_path)
    orig_image = skimage.img_as_float(skimage.io.imread(image_path, as_grey=False)).astype(np.float32)
    padding = 0
    results = segment(orig_image, pad=padding)

    results_dir = os.path.join(os.getcwd(), "plots", ("padding_test_" + str(padding) + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    os.mkdir(results_dir)

    # minc_plot.plot_probability_maps(av_prob_maps, results)
    minc_plot.plot_class_map(results, save=True, path=results_dir)
    minc_plot.plot_confidence_map(results,save=True, path=results_dir)

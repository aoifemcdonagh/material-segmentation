# Script containing functions for performing image segmentation on desktop environment with GPU

import numpy as np
from PIL import Image
import skimage
import time
import os
os.environ['GLOG_minloglevel'] = '2'  # Suppressing caffe printouts of network initialisation
import caffe
caffe.set_mode_gpu()

top_dir = os.path.dirname(os.path.realpath(__file__))  # Find path to directory containing this script
default_caffemodel = top_dir+"/../models/minc-googlenet-conv.caffemodel" # default caffemodel file

SCALES = [1.0 / np.sqrt(2), 1.0, np.sqrt(2)]  # Define scales as per MINC paper

#  Global dictionary containing class number : name pairs
CLASS_LIST = {0: "brick",
              1: "carpet",
              2: "ceramic",
              3: "fabric",
              4: "foliage",
              5: "food",
              6: "glass",
              7: "hair",
              8: "leather",
              9: "metal",
              10: "mirror",
              11: "other",
              12: "painted",
              13: "paper",
              14: "plastic",
              15: "polishedstone",
              16: "skin",
              17: "sky",
              18: "stone",
              19: "tile",
              20: "wallpaper",
              21: "water",
              22: "wood"}


def segment(im, pad=0, caffemodel=None):
    """
    Function which segments an input image. uses pyramidal method of scaling, performing
    inference, upsampling results, and averaging results.
    :param im: image to segment
    :param pad: number of pixels of padding to add
    :param caffemodel: path to caffemodel file
    :return: The upsampled and averaged results of inference on input image at 3 scales.
    """
    caffe.set_mode_gpu()

    padded_image = add_padding(im, pad)  # Add padding to original image
    resized_images = resize_images(padded_image)  # Resize original images

    outputs = [classify(image, caffemodel=caffemodel) for image in resized_images]  # Perform classification on images

    upsample_start = time.time()
    average_prob_maps = get_average_prob_maps(outputs, im.shape, pad)
    print("Total segmenting time: {:.3f} ms".format((time.time() - upsample_start) * 1000))

    return average_prob_maps


def get_average_prob_maps(network_outputs, shape, pad=0):
    """
    :param network_outputs: List of outputs
    :param shape: shape of original image needed for upsampling
    :return: Probability maps for each class, averaged from resized images probability maps
    """

    # Get probability maps for each class for each image
    prob_maps = [get_probability_maps(out) for out in network_outputs]

    # Upsample probability maps to dimensions of original image (plus any padding)
    # Output shape in (width, height) format for PIL.Image resizing.
    # Swap shape[1] and shape[0] if using skimage resizing
    upsampled_prob_maps = upsample_PIL(prob_maps, output_shape=(shape[1] + pad*2, shape[0] + pad*2))

    # Average probability maps for each class.
    # Average across upsampled inference results of all resized images
    averaged_prob_maps = np.average(upsampled_prob_maps, axis=0)

    # Remove the padded sections from the averaged prob maps
    averaged_prob_maps = [remove_padding(prob_map, pad) for prob_map in averaged_prob_maps]

    return np.array(averaged_prob_maps)


# This function not used anymore. Here for reference
def upsample(prob_maps_multiple_images, output_shape):
    """
    Function for performing upsamping of probability maps
    :param prob_maps: Probability maps for each class for each resized image
    :param output_shape: Desired shape to upsample to (should be dimensions of original image)
    :return:
    """
    upsample_start = time.time()
    # Upsampling probability maps to be same dimensions as original image
    # np.array so that they can be averaged later
    upsampled = np.array([[skimage.transform.resize(prob_map,
                                               output_shape=output_shape,
                                               mode='constant',
                                               cval=0,
                                               preserve_range=True)
                      for prob_map in prob_maps_single_image]
                    for prob_maps_single_image in prob_maps_multiple_images])
    print("upsample time: {:.3f} ms".format((time.time() - upsample_start) * 1000))
    return upsampled


def upsample_PIL(prob_maps_multiple_images, output_shape):
    """
    Function for performing upsamping of probability maps
    USES PIL RESIZE FUNCTION INSTEAD OF SKIMAGE
    :param prob_maps: Probability maps for each class for each resized image
    :param output_shape: Desired shape to upsample to (should be dimensions of original image)
    :return:
    """
    upsample_start = time.time()
    # Upsampling probability maps to be same dimensions as original image
    # np.array so that they can be averaged later
    upsampled = np.array([[np.array(Image.fromarray(prob_map).resize(size=output_shape, resample=Image.BILINEAR))
                      for prob_map in prob_maps_single_image]
                    for prob_maps_single_image in prob_maps_multiple_images])

    print("upsample time PIL: {:.3f} ms".format((time.time() - upsample_start) * 1000))
    return upsampled


def resize_images(im):
    """
    Function for resizing an image to scales as defined in MINC paper
    images are not saved, instead returned for processing

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


def classify(im, caffemodel=None):
    """
    Function performing material classification across a whole image of arbitrary size.
    This function can be called if no upsampling is desired

    :param im: image preloaded using caffe.io.load_image()
    :param caffemodel: name of .caffemodel file
    :return: network output
    """

    if caffemodel is None:  # If no caffemodel file specified use default
        caffemodel = default_caffemodel

    # Prototxt file is assumed to have the same name as caffemodel file
    prototxt = os.path.splitext(caffemodel)[0] + ".prototxt"

    net_full_conv = caffe.Net(prototxt, caffemodel, caffe.TEST)  # Load network
    net_full_conv.blobs['data'].reshape(1, 3, im.shape[0], im.shape[1])  # Reshape the input layer to image size

    transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([104, 117, 124]))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    inf_start = time.time()
    # make classification map by forward and print prediction indices at each location
    out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    print("Inference time: {:.3f} ms".format((time.time() - inf_start)*1000))

    return out


def get_probability_maps(network_output):
    """
    Function which returns all probability maps in a network output
    Returns a list of probability maps (numpy arrays)
    """
    return [network_output['prob'][0][class_num] for class_num in CLASS_LIST.keys()]


def get_class_map(network_output):
    """
    function taking network output and returning a map of highest probability classes at each location
    Map of class names
    Used for estimating average absorption coeff?
    :param network_output:
    :return:
    """

    return network_output['prob'][0].argmax(axis=0)  # Get highest probability class at each location

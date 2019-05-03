# Script for performing pyramidal upsampling and segmentation on NCS
# Segments a single image
# Very slow since network has to be loaded every time it's reshaped, i.e. 3 times per segmented image.
# This script implements own segment function customised for NCS
import sys
import os
import cv2
import skimage.transform
from openvino.inference_engine import IENetwork, IEPlugin
import logging as log
import numpy as np
from time import time
import argparse
import minc_plotting as minc_plot
import ncs_demos.ncs_plotting as ncs_plot

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

SCALES = [1.0 / np.sqrt(2), 1.0, np.sqrt(2)]  # changed scales


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--image", help="Path to a single image file", required=True,
                        type=str)
    parser.add_argument("-p", "--padding", help="Number of pixels of padding to add", type=int, default=0)
    return parser


def segment(network_outputs, im, pad=0):
    """
    Function which takes a list of network outputs and returns an upsampled classification map suitable for plotting
    :param network_outputs: list of network outputs
    :return: upsampled classification map
    """

    return get_average_prob_maps(network_outputs, im.shape, pad)



def get_average_prob_maps(network_outputs, shape, pad=0):
    """
    :param shape: shape of original image needed for upsampling
    :param network_outputs: List of outputs from
    :return: Probability maps for each class, averaged from resized images probability maps
    """

    # Get probability maps for each class for each image
    prob_maps = [get_probability_maps(out) for out in network_outputs]

    # Upsample probability maps to dimensions of original image (plus any padding)
    upsampled_prob_maps = upsample_prob_maps(prob_maps, output_shape=(shape[1] + pad*2, shape[2] + pad*2))

    # Probability maps for each class, averaged from resized images probability maps
    averaged_prob_maps = np.average(upsampled_prob_maps, axis=0)

    # Remove the padded sections from the averaged prob maps
    averaged_prob_maps = [remove_padding(prob_map, pad) for prob_map in averaged_prob_maps]

    return np.array(averaged_prob_maps)


def upsample_prob_maps(prob_maps_multiple_images, output_shape):
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


def resize_image(im):
    """
    Resize input image to 3 scales as defined in MINC paper
    :param image: image with float (16 or 32) values in range [0-255]
    :return: List of 3 resized images
    """

    im = im/255.0  # Images of type float must be between -1 and 1 to perform rescaling
    return [skimage.transform.rescale(im, scale, mode='constant', cval=0)*255.0 for scale in SCALES]


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


def get_probability_maps(network_output):
    """
    Function which returns all probability maps in a network output
    Returns a list of probability maps (numpy arrays)
    """
    return [network_output['prob'][0][class_num] for class_num in CLASS_LIST.keys()]


def preprocess_image(im, pad=0):
    """
    Pre-process input image
    :param im: loaded image
    :param pad: number of pixels to pad
    :return: list of padded/resized/transposed images ready for inference with MINC model
    """
    padded_image = add_padding(im, pad=pad)
    images = resize_image(padded_image)  # perform resizing before transposing
    images = [image.transpose((2, 0, 1)) for image in images]  # Change data layout from HWC to CHW

    return images


if __name__ == "__main__":
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)  # Configure logging
    args = build_argparser().parse_args()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for Movidius stick
    plugin = IEPlugin(device="MYRIAD")

    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.image)  # Should be 1

    # Read and pre-process input images
    # Image loaded as type float32. Works as expected with NCS
    # float16 was thought to be required by NCS but skimage.transform.rescale throws error for this type.
    im = cv2.imread(args.image).astype(np.float32)
    processed_images = preprocess_image(im, pad=args.padding)
    results = []

    for image in processed_images:
        # Reshape input layer for image
        net.reshape({input_blob: (1, image.shape[0], image.shape[1], image.shape[2])})

        # Loading model to the plugin
        # Model needs to be loaded every time network input is resized.
        log.info("Loading model to the plugin")
        exec_net = plugin.load(network=net)  # Loading network multiple times takes a long time

        # Start sync inference
        log.info("Starting inference ")
        t0 = time()
        results.append(exec_net.infer(inputs={input_blob:image}))
        log.info("Average running time of one iteration: {} ms".format((time() - t0) * 1000))

    log.info("processing output blob")

    segmented_results = segment(results, processed_images[1], pad=args.padding)

    minc_plot.plot_class_map(segmented_results)

    ncs_plot.plot_class_map(segmented_results)

    log.info("done")

    #output = result[out_blob]
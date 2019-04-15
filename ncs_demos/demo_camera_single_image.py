import sys
import os
import cv2
import skimage.transform
from openvino.inference_engine import IENetwork, IEPlugin
import logging as log
import numpy as np
from time import time
import argparse
import segment as minc_utils


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
    parser.add_argument("-i", "--input", help="Input, 'cam' or path to image", required=True,
                        type=str)
    parser.add_argument("-p", "--padding", help="Number of pixels of padding to add", type=int)
    parser.add_argument("-r", "--resolution", help="desired camera resolution, format [h,w]", nargs="+", type=int)
    return parser


def generate_class_map(network_outputs, shape, pad=0):
    """
    Function which takes a list of network outputs and returns an upsampled classification map suitable for plotting
    :param network_outputs: list of network outputs
    :return: upsampled classification map
    """

    av_prob_maps = minc_utils.get_average_prob_maps(network_outputs, shape, pad)
    return av_prob_maps.argmax(axis=0)


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

    log.info("resolution = {} {}".format(args.resolution[0], args.resolution[1]))
    net.reshape({input_blob: (1, 3, args.resolution[0], args.resolution[1])})

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)  # Loading network multiple times takes a long time

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    cur_request_id = 0
    next_request_id = 1

    log.info("Starting inference in async mode...")
    log.info("To switch between sync and async modes press Tab button")
    log.info("To stop the demo execution press Esc button")
    is_async_mode = True
    render_time = 0
    ret, frame = cap.read()

    while cap.isOpened():
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
        if not ret:
            break

        if is_async_mode:
            in_frame = cv2.resize(next_frame, (args.resolution[1], args.resolution[0]))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((1, 3, args.resolution[0], args.resolution[1]))
            exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        else:
            in_frame = cv2.resize(frame, (args.resolution[1], args.resolution[0]))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((1, 3, args.resolution[0], args.resolution[1]))
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})

        if exec_net.requests[cur_request_id].wait(-1) == 0:
            results = exec_net.requests[cur_request_id].outputs[out_blob]
            class_map = generate_class_map(results, [args.resolution[0], args.resolution[1]], pad=args.padding)



        cv2.imshow("Segmentation results", class_map)

    log.info("done")
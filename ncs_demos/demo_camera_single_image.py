import sys
import os
import cv2
from openvino.inference_engine import IENetwork, IEPlugin
import logging as log
import argparse
sys.path.insert(0, "../")  # add sys path to find ncs demos
import material_segmentation.plotting_utils as utils


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input", help="Input, 'cam' or path to image", required=True,
                        type=str)
    parser.add_argument("-p", "--padding", help="Number of pixels of padding to add", type=int)
    parser.add_argument("-r", "--resolution", help="desired camera resolution, format [h,w]", nargs="+", type=int)
    return parser


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

    n = 1
    c = 3
    h = args.resolution[0]
    w = args.resolution[1]

    log.info("resolution = {} {}".format(h, w))
    net.reshape({input_blob: (n, c, h, w)})

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

    ret, frame = cap.read()

    # Transforming image before inference
    # Note that BGR to RGB is not performed because network assumes BGR format anyway
    in_frame = cv2.resize(frame, (w, h))  # Does resize() crop or shrink image dimensions?
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))  # adding n dimension
    exec_net.infer(inputs={input_blob: in_frame})

    # Parse detection results of the current request
    result = exec_net.requests[cur_request_id].outputs
    av_prob_maps = utils.get_average_prob_maps([result], [h,w])  # Performing upsampling
    class_map = utils.get_pixel_map(av_prob_maps, )

    cv2.imshow('class map', class_map)
    key = cv2.waitKey()

    del net
    del exec_net
    del plugin

    log.info("done")

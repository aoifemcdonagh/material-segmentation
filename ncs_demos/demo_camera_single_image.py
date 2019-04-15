import sys
import os
import cv2
from openvino.inference_engine import IENetwork, IEPlugin
import logging as log
import argparse
from ncs_demos.ncs_utilities import get_average_prob_maps


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

    av_prob_maps = get_average_prob_maps(network_outputs, shape, pad)
    return av_prob_maps.argmax(axis=0)


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
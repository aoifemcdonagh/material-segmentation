import sys
import os
import cv2
from openvino.inference_engine import IENetwork, IEPlugin
import logging as log
import argparse
sys.path.insert(0, "../")  # add sys path to find ncs demos
import ncs_demos.ncs_utilities as utils


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input", help="Input, 'cam' or path to image", required=True,
                        type=str)
    parser.add_argument("-u", "--upsample", help="To upsample output", default=False, action="store_true")
    parser.add_argument("-p", "--padding", help="Number of pixels of padding to add", type=int, default=0)
    parser.add_argument("-r", "--resolution", help="desired camera resolution, format [h,w]", nargs="+", type=int)
    return parser


def generate_class_map(network_outputs, shape, pad=0):
    """
    Function which takes a list of network outputs and returns an upsampled classification map suitable for plotting
    :param network_outputs: list of network outputs
    :return: upsampled classification map
    """

    av_prob_maps = utils.get_average_prob_maps([network_outputs], shape, pad)
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

    n = 1
    c = 3
    h = args.resolution[0]
    w = args.resolution[1]

    log.info("resolution = {} {}".format(h, w))
    net.reshape({input_blob: (n, c, h, w)})

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net, num_requests=2)

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

    window_result = cv2.namedWindow("class map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("class map", h, w)

    window_frame = cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
        if not ret:
            break
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        if is_async_mode:
            cv2.imshow("frame", next_frame)
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        else:
            cv2.imshow("frame", frame)
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            # Parse detection results of the current request
            result = exec_net.requests[cur_request_id].outputs
            if args.upsample:
                av_prob_maps = utils.get_average_prob_maps([result], [h,w], pad=args.padding)
                class_map = utils.get_class_map(av_prob_maps)
            else:
                class_map = utils.get_class_map(result)

            cv2.imshow('class map', class_map)

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame

        key = cv2.waitKey(10)
        if key == 27:
            break
        if (9 == key):
            is_async_mode = not is_async_mode
            log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))


    del net
    del exec_net
    del plugin

    log.info("done")

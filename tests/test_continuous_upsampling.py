import segment
import ncs_demos.ncs_utilities as utils
import cv2
import logging as log
import argparse
import sys
import os
import caffe


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", type=str)
    parser.add_argument("-i", "--input", help="Input, 'cam' or path to image", required=True,
                        type=str)
    parser.add_argument("-u", "--upsample", help="To upsample output", default=False, action="store_true")
    parser.add_argument("-p", "--padding", help="Number of pixels of padding to add", type=int, default=0)
    parser.add_argument("-r", "--resolution", help="desired camera resolution, format [h,w]", nargs="+", type=int)
    return parser


if __name__ == "__main__":
    caffe.set_mode_gpu()
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)  # Configure logging
    args = build_argparser().parse_args()

    n = 1
    c = 3
    h = args.resolution[0]
    w = args.resolution[1]

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    window_result = cv2.namedWindow("class map", cv2.WINDOW_NORMAL)
    window_frame = cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()

        results = segment.segment(frame, args.padding)
        class_map = utils.get_class_map(results)

        cv2.imshow('class map', class_map)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)
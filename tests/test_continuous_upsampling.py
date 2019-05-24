# This script performs continuous segmentation with pyramidal upsampling on video/camera stream

import bonaire.plotting_utils as utils
from bonaire.gpu_segment import segment
import cv2
import logging as log
import argparse
import os
import sys


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file of a trained model.", type=str)
    parser.add_argument("-i", "--input", help="Input, 'cam' or path to image", required=True, type=str)
    parser.add_argument("-p", "--padding", help="Number of pixels of padding to add", type=int, default=0)
    #parser.add_argument("-r", "--resolution", help="desired camera resolution, format [h,w]", nargs="+", type=int)
    return parser


if __name__ == "__main__":
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)  # Configure logging
    args = build_argparser().parse_args()

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    window_result = cv2.namedWindow("class map")
    window_frame = cv2.namedWindow("frame")

    while cap.isOpened():
        ret, frame = cap.read()
        # convert current frame to RGB for processing by 'gpu_segment'
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = segment(RGBframe, args.padding, args.model)
        class_map = utils.get_pixel_map(results)

        cv2.imshow('class map', class_map)
        cv2.imshow('frame', frame)
        cv2.waitKey(10)

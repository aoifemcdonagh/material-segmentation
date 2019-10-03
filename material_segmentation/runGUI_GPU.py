from material_segmentation.SegmentationApp import SegmentationApp
import logging as log
import caffe
import sys
import argparse
import os
import time
import cv2


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to a .caffemodel file with a trained model.", type=str)
    parser.add_argument("-i", "--input", help="Input, 'cam' or path to image", required=True,
                        type=str)
    #parser.add_argument("-u", "--upsample", help="To upsample output", default=False, action="store_true")
    parser.add_argument("-p", "--padding", help="Number of pixels of padding to add", type=int, default=0)
    #parser.add_argument("-r", "--resolution", help="desired camera resolution, format [h,w]", nargs="+", type=int)
    return parser


if __name__ == '__main__':
    caffe.set_mode_gpu()
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)  # Configure logging
    args = build_argparser().parse_args()

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    log.info("Setting up camera")

    video = cv2.VideoCapture(input_stream)
    #time.sleep(2.0)

    # start the app
    sa = SegmentationApp(video, args.padding, args.model)
    sa.root.mainloop()

# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tk
import threading
import argparse
import logging as log
import caffe
import sys
import datetime
import imutils
import cv2
import os
import time


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", type=str)
    parser.add_argument("-i", "--input", help="Input, 'cam' or path to image", required=True,
                        type=str)
    parser.add_argument("-u", "--upsample", help="To upsample output", default=False, action="store_true")
    parser.add_argument("-p", "--padding", help="Number of pixels of padding to add", type=int, default=0)
    parser.add_argument("-r", "--resolution", help="desired camera resolution, format [h,w]", nargs="+", type=int)
    return parser


class SegmentationApp:
    def __init__(self, vs):
        self.vs = vs  # video stream
        self.frame = None
        self.thread = None
        self.stopEvent = None

        self.root = tk.Tk()
        self.panel = None

        # create a button, that when pressed, will start segmentation on video stream
        btn = tk.Button(self.root, text="Snapshot!", command=self.start_segmentation)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

        # Start a thread which reads from camera/video file
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def video_loop(self):
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream
                self.frame = self.vs.read()
                #self.frame = imutils.resize(self.frame, width=300)  # May need to resize??

                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                # image processing is done by caffe before feeding to CNN

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tk.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError:
                print("[INFO] caught a RuntimeError")

    def start_segmentation(self):
        """
        Function for
        :return:
        """

    def on_close(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()


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

    vs = cv2.VideoStream(input_stream).start()
    time.sleep(2.0)

    # start the app
    sa = SegmentationApp(vs)
    sa.root.mainloop()
# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tk
import threading
from bonaire.gpu_segment import segment
import bonaire.plotting_utils as utils
import cv2


class SegmentationApp:
    def __init__(self, vc, pad):
        self.vc = vc  # video capture
        self.padding = pad
        self.frame = None
        self.video_thread = None
        self.segment_thread = None
        self.stopVideo = None
        self.stopSegmenting = None

        self.root = tk.Tk()
        self.panel = None

        # create a button, that when pressed, will start segmentation on video stream
        btn = tk.Button(self.root, text="segment!", command=self.segment)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

        # Start a thread which reads from camera/video file
        self.stopVideo = threading.Event()
        self.video_thread = threading.Thread(target=self.video_loop, args=())
        self.video_thread.start()

        # Start thread which performs segmentation
        #self.stopSegmenting = threading.Event()
        #self.segment_thread = threading.Thread(target=self.segment, args=())
        #self.segment_thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)


    def video_loop(self):
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopVideo.is_set():
                # grab the frame from the video stream
                _, frame = self.vc.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # image processing is done by caffe before feeding to CNN
                results = segment(frame, pad=self.padding)
                class_map = utils.get_pixel_map(results)
                class_map = Image.fromarray(class_map)
                class_map = ImageTk.PhotoImage(class_map)

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tk.Label(image=class_map)
                    self.panel.image = class_map
                    self.panel.pack(side="left", padx=10, pady=10)

                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=class_map)
                    self.panel.image = class_map

        except RuntimeError:
                print("[INFO] caught a RuntimeError")

    def segment(self):
        """
        Function for
        :return:
        """
        print("Segmenting")


    def on_close(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()

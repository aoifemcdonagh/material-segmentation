# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tk
import threading
from bonaire.gpu_segment import segment
import ncs_demos.ncs_utilities as utils
import cv2


class SegmentationApp:
    def __init__(self, vc, pad):
        self.vc = vc  # video capture
        self.padding = pad
        self.frame = None
        self.camera_thread = None
        self.stopPlotting = None
        self.stopSegmenting = None

        self.root = tk.Tk()
        self.panel = None

        # create a button, that when pressed, will start segment a video stream frame
        btn_segment = tk.Button(self.root, text="segment!", command=self.start_segment)
        btn_segment.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

        # Button to start video stream display
        btn_start_display = tk.Button(self.root, text="start video", command=self.start_video)
        btn_start_display.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

        # Start a thread which reads from camera/video file and displays frames
        self.stopVideo = threading.Event()
        self.video_thread = threading.Thread(target=self.start_video, args=())
        self.video_thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("Material Segmentation")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)

    def video_loop(self):
        """
        Function which displays frames from video stream
        :return:
        """
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopVideo.is_set():
                # grab the frame from the video stream
                _, self.frame = self.vc.read()
                print("reading frames")
                frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  # Convert to RGB before plotting
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tk.Label(image=frame)
                    self.panel.image = frame
                    self.panel.pack(side="left", padx=10, pady=10)

                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=frame)
                    self.panel.image = frame
                    print("put up new frame")

        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def segment(self):
        """
        Function for performing segmentation
        :return:
        """
        try:
            # convert current frame to RGB for processing by 'gpu_segment'
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # image processing is done by caffe before feeding to CNN
            results = segment(frame, pad=self.padding)

            # Plot results
            class_map = utils.get_class_map(results)
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

    def start_video(self):
        self.stopVideo.clear()  # Clear stop video flag
        self.video_loop()

    def start_segment(self):
        self.stopVideo.set()  # Stop reading from video stream and segment current frame
        self.segment()

    def on_close(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopVideo.set()
        self.vc.stop()
        self.root.quit()

# This script is based off SegmentationApp.py. Modified to perform inference on NCS

from PIL import Image
from PIL import ImageTk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import material_segmentation.plotting_utils as utils
from ncs_demos import ncs_segment
import cv2
import os
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin


class SegmentationApp:
    def __init__(self, vc, model_path, pad):
        self.vc = vc  # video capture
        self.padding = pad
        self.model = model_path
        self.frame = None
        self.stopVideo = None  # Flag for stopping video loop
        self.colorbar_frame = None

        self.root = tk.Tk()
        self.panel = None
        self.colorbar_panel = None
        self.resolution = [500, 700]

        button_frame = tk.Frame()
        button_frame.pack(side="bottom")

        # create a button, that when pressed, will show current frame
        btn_frame = tk.Button(button_frame, text="Show Frame", command=self.show_frame)
        btn_frame.pack(side="left", padx=10, pady=10)

        # create a button, that when pressed, will start segment a video stream frame
        btn_segment = tk.Button(button_frame, text="Class map", command=self.segment_classes)
        btn_segment.pack(side="left",  padx=10, pady=10)

        # button for segmenting and showing absorption coefficient map
        btn_abs = tk.Button(button_frame, text="Heat map", command=self.segment_heatmap)
        btn_abs.pack(side="left", padx=10, pady=10)

        # Button to start video stream display
        btn_start_display = tk.Button(button_frame, text="start video", command=self.start_video)
        btn_start_display.pack(side="right", padx=10, pady=10)

        # set a callback to handle when the window is closed
        self.root.wm_title("Material Segmentation")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)

        # Set up NCS
        # Want to set up model once so that it doesn't have to be loaded multiple times
        model_xml = model_path
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Plugin initialization for Movidius stick
        plugin = IEPlugin(device="MYRIAD")

        # Read IR
        net = IENetwork(model=model_xml, weights=model_bin)

        self.input_blob = next(iter(net.inputs))

        n = 1
        c = 3
        h = self.resolution[0]
        w = self.resolution[1]

        net.reshape({self.input_blob: (n, c, h, w)})

        # Loading model to the plugin
        self.exec_net = plugin.load(network=net)  # Loading network multiple times takes a long time

    def video_loop(self):
        """
        Function which displays frames from video stream
        :return:
        """

        # keep looping over frames until we are instructed to stop
        while not self.stopVideo.is_set():
            # grab the frame from the video stream
            _, self.frame = self.vc.read()

            # If the frame is too large to plot, make it smaller.
            if self.frame.shape[0] > 500 or self.frame.shape[1] > 800:
                self.frame = cv2.resize(self.frame, (1280, 720), interpolation=cv2.INTER_AREA)

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

    def segment(self, map_type):
        """
        Function for performing and plotting segmentation
        :return:
        """

        # Check if there are any Canvas objects in GUI children and destroy them
        for child in list(self.root.children.values()):
            if child.widgetName == 'canvas':
                child.destroy()

        try:
            # convert current frame to RGB for processing by 'gpu_segment'
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            in_frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]))  # Does resize() crop or shrink image dimensions?
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((1, 3, self.resolution[0], self.resolution[1]))  # adding n dimension
            self.exec_net.infer(inputs={self.input_blob: in_frame})

            # Parse detection results of the current request
            results = self.exec_net.requests[0].outputs
            results = utils.get_average_prob_maps_single_image(results, [self.resolution[0], self.resolution[1]])

            engine = utils.PlottingEngine()
            engine.set_colormap(map_type=map_type, freq=1000)  # Have colormap set by button in GUI in future

            pixels, colorbar, _ = engine.process(results)

            image = Image.fromarray(pixels)
            image = ImageTk.PhotoImage(image)

            colorbar = FigureCanvasTkAgg(colorbar, master=self.root)
            colorbar.get_tk_widget().pack(side="right", padx=10, pady=10)

            # Update panel with segmented image.
            self.panel.configure(image=image)
            self.panel.image = image

            self.root.update_idletasks()

        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def show_frame(self):
        """
        Function which will show the current frame in GUI
        :return:
        """
        try:
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)

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

    def start_video(self):
        # Check if there are any Canvas objects in GUI children and destroy them
        for child in list(self.root.children.values()):
            if child.widgetName == 'canvas':
                child.destroy()

        # Start a thread which reads from camera/video file and displays frames
        self.stopVideo = threading.Event()
        self.video_thread = threading.Thread(target=self.video_loop, args=())
        self.video_thread.start()

    def segment_classes(self):
        self.stopVideo.set()  # Stop reading from video stream and segment current frame
        self.segment(map_type="classes")

    def segment_heatmap(self):
        self.stopVideo.set()  # Stop reading from video stream and segment current frame
        self.segment(map_type="absorption")

    def on_close(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopVideo.set()
        self.vc.release()
        self.root.destroy()

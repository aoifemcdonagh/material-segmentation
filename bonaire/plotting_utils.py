# Contains functions for modifying classification results to be plotted

import skimage.transform
import numpy as np
import logging as log
import time
import sys
import csv
import matplotlib.pyplot as plt
import cv2

abs_coeff_file = "/home/aoife/projects/bonaire/abs_coefficients.csv"

#  Global dictionary containing class number : name pairs
CLASS_LIST = {0: "brick",
              1: "carpet",
              2: "ceramic",
              3: "fabric",
              4: "foliage",
              5: "food",
              6: "glass",
              7: "hair",
              8: "leather",
              9: "metal",
              10: "mirror",
              11: "other",
              12: "painted",
              13: "paper",
              14: "plastic",
              15: "polishedstone",
              16: "skin",
              17: "sky",
              18: "stone",
              19: "tile",
              20: "wallpaper",
              21: "water",
              22: "wood"}

SCALES = [1.0 / np.sqrt(2), 1.0, np.sqrt(2)]  # Define scales as per MINC paper


class PlottingEngine:
    def __init__(self):
        self.colormap = None  # Colormap is always RGB. Keep this in mind when using OpenCV which assumes BGR format.

    def process(self, network_output):
        """
        Function which returns an image and a colorbar to be shown in Tkinter GUI
        :param results: Network output
        :param color_map: colormap
        :return:
        """

        if self.colormap is None:  # Initialise a colormap if none was set.
            self.set_colormap("classes")

        # Get class map. used to generate pixel map and color bar.
        if type(network_output) is dict:  # If the input value is an unmodified 'network output' (from a single image)
            class_map = network_output['prob'][0].argmax(axis=0)  # Get highest probability class at each location
        else:  # if average probability maps are passed in in case of upsampling & averaging
            class_map = network_output.argmax(axis=0)  # Get highest probability class at each location

        pixel_map = self.get_pixel_map(class_map)

        colorbar = self.create_colorbar(class_map)

        return pixel_map, colorbar

    def get_pixel_map(self, class_map):
        """
        Function to generate a pixel map (from network output) which can be plotted by OpenCV
        :param network_output: output from network (inference on GPU or NCS)
        :return: an array of tuples to be plotted by OpenCV. The tuples define pixel values
        """

        # Convert to format suitable for plotting with OpenCV, i.e. array of pixels
        pixel_map = np.array([[self.colormap[class_num] for class_num in row] for row in class_map], dtype=np.uint8)
        return pixel_map

    def set_colormap(self, map_type, freq=None):
        """
        Set desired colormap for plotting
        :param map_type: string specifying what type of colormap to use
        :param freq: int denoting frequency band for which to create color map based on material absorption coeffs
        :return:
        """

        if map_type == "absorption":
            cmap = self.generate_grayscale_map(freq)
            print("Colormap = absorption")

        elif map_type == "classes":
            cmap = self.generate_color_map()
            print("Colormap = classes")

        else:
            print("Invalid map choice")
            return

        self.colormap = cmap

    def generate_color_map(self):
        """
        Method which generates evenly spaced colors for plotting classes

        Scope here to make dynamic colormap which changes based on how many
        classes are present in a frame.
        :return:
        """
        hues = np.linspace(0, 179, num=len(CLASS_LIST), dtype=int)
        hsv_colors = []
        for hue in hues:
            hsv_colors.append((hue, 255, 255))

        rgb_colors = []
        for hsv_color in hsv_colors:
            rgb_colors.append(cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2RGB)[0][0])

        return rgb_colors

    def generate_grayscale_map(self, band, file_path=abs_coeff_file):
        """
        Returns a map of scalar values based on given absorption coefficients.
        Output can be used as greyscale colormap for plotting segmentation results
        :param: band: frequency band to get absorption coefficients from [125,250,500,1000,2000,4000]
        :param file_path: path to file containing absorption coefficients
        :return: Grayscale map based on absorption coefficients at a given freq band
        """
        band = 4000 if band is None else band  # Set band to 4000 Hz if no value set.

        with open(file_path) as csvfile:
            read_csv = csv.reader(csvfile, delimiter=',')
            headers = next(read_csv, None)  # Get headers in csv file

            if str(band) not in headers:
                print("Invalid frequency band")
                return
            else:
                index = headers.index(str(band))

            abs_coeffs = []
            for row in read_csv:
                abs_coeffs.append(float(row[index]))  # Get abs coeff for each material at given freq band as float

            # absorption coefficients interpolated between [0, 255]
            #interpolated_coeffs = np.interp(abs_coeffs, [min(abs_coeffs), max(abs_coeffs)], [0, 255]).astype(int)
            interpolated_coeffs = np.interp(abs_coeffs, [0, 1], [0, 255]).astype(int)

            hsv_colors = []
            for coeff in interpolated_coeffs:
                hsv_colors.append((0, 0, (255 - coeff)))

            rgb_colors = []
            for hsv_color in hsv_colors:
                rgb_colors.append(cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2RGB)[0][0])

            return rgb_colors

    def create_colorbar(self, class_map):
        """
        Function for creating a plot of colorbar to display beside segmentation results.
        Creates colorbar for values in self.colormap
        :return:
        """

        print("Creating colorbar using color map: " + str(self.colormap))

        #modified_values = range(0, len(unique_values))  # list in range 0 - len(unique_values)
        #value_dict = {a: b for (a, b) in zip(unique_values, modified_values)}

        #unique_values = np.arange(0,23)
        unique_values = np.unique(class_map)  # array of unique values in class_map
        print(unique_values)
        b = np.ones([len(unique_values), 4])
        bar = (b * unique_values[:, np.newaxis]).astype(int)

        colorbar_pixels = self.get_pixel_map(bar)

        # Creating a color bar to display
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        # Create the colormap
        # Get color map in range [0, 1]
        #color_map = [[x / 255 for x in rgb_tuple] for rgb_tuple in self.colormap]
        # number of bins is the number of classes
        #cm = LinearSegmentedColormap.from_list("minc_material_map", color_map, N=23)
        ax.imshow(colorbar_pixels)
        labels = self.get_tick_labels(unique_values)
        ax.set_yticks(np.arange(0, len(unique_values)))
        ax.set_yticklabels(labels)

        return fig

    """
    def create_opencv_colorbar(self, class_map):
        unique_values = np.unique(class_map)  # array of unique values in class_map
        unique_values = np.repeat(unique_values, 50)
        b = np.ones([len(unique_values), 50])
        bar = (b * unique_values[:, np.newaxis]).astype(int)

        # Convert bar of class nums to pixel values
        pixel_map = np.array([[self.colormap[class_num] for class_num in row] for row in bar], dtype=np.uint8)

        # Need to convert map of pixels to image.
        image = Image.fromarray(pixel_map)
        image = ImageTk.PhotoImage(image)

        return image
    """

    def get_tick_labels(self, class_numbers):
        """
        Function generating tick labels appropriate to each classified image
        """
        class_names = []
        for number in class_numbers:
            class_names.append(CLASS_LIST.get(number))

        tick_labels = []
        for (number, name) in zip(class_numbers, class_names):
            tick_labels.append(str(number) + ": " + name)

        return tick_labels


"""
Following functions are used by NCS demo scripts

get_average_prob_maps is used in demo_camera.py to upsample on one image instead of 3
"""


def get_pixel_map(class_map, colormap):
    """
    Function to generate a pixel map (from network output) which can be plotted by OpenCV
    :param network_output: output from network (inference on GPU or NCS)
    :return: an array of tuples to be plotted by OpenCV. The tuples define pixel values
    """

    # Convert to format suitable for plotting with OpenCV, i.e. array of pixels
    pixel_map = np.array([[colormap[class_num] for class_num in row] for row in class_map], dtype=np.uint8)
    return pixel_map

def get_average_prob_maps(network_outputs, shape, pad=0):
    """
    :param network_outputs: List of outputs
    :param shape: shape of original image needed for upsampling
    :return: Probability maps for each class, averaged from resized images probability maps
    """
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)  # Configure logging
    # Get probability maps for each class for each image
    prob_maps = [get_probability_maps(out) for out in network_outputs]

    upsample_start = time.time()
    # Upsample probability maps to dimensions of original image (plus any padding)
    upsampled_prob_maps = upsample(prob_maps, output_shape=(shape[0] + pad*2, shape[1] + pad*2))
    log.info("Upsampling operation time: {:.3f} ms".format((time.time() - upsample_start) * 1000))

    # Probability maps for each class, averaged from resized images probability maps
    # Averaging over axis 0 removes the 'n' dimension so that the output is 23*H*W
    averaged_prob_maps = np.average(upsampled_prob_maps, axis=0)

    # Remove the padded sections from the averaged prob maps
    averaged_prob_maps = np.array([remove_padding(prob_map, pad) for prob_map in averaged_prob_maps])

    return averaged_prob_maps


def upsample(prob_maps_multiple_images, output_shape):
    """
    Function for performing upsamping of probability maps
    :param prob_maps: Probability maps for each class for each resized image
    :param output_shape: Desired shape to upsample to (should be dimensions of original image)
    :return:
    """

    # Upsampling probability maps to be same dimensions as original image
    # np.array so that they can be averaged later
    return np.array([[skimage.transform.resize(prob_map,
                                               output_shape=output_shape,
                                               mode='constant',
                                               cval=0,
                                               preserve_range=True)
                      for prob_map in prob_maps_single_image]
                    for prob_maps_single_image in prob_maps_multiple_images])


def resize_images(im):
    """
    Function for resizing an image to scales as defined in MINC paper
    Not saving images anymore, just returning them for processing

    :param im: pre-loaded image
    :return: list of resized images
    """

    return [skimage.transform.rescale(im, scale, mode='constant', cval=0) for scale in SCALES]


def get_probability_maps(out):
    """
    Function which returns all probability maps in a network output
    :param out: network output
    :return: list of probability maps (numpy arrays)
    """

    return [out['prob'][0][class_num] for class_num in CLASS_LIST.keys()]


def add_padding(im, pad=0):
    """
    Function for padding image before classification
    :param im: image (preloaded with caffe.io.load_image)
    :param pad: number of pixels of padding to add (default 0)
    :return: image with padding
    """

    return np.pad(im, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='symmetric')


def remove_padding(im, pad=0):
    """
    Function for removing padding from an image
    :param im: image or prob map to remove padding from
    :param pad: number of pixels of padding to remove
    :return:
    """

    if pad == 0:
        return im
    else:
        return im[pad:-pad, pad:-pad]

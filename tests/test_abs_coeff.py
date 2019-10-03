"""
Script for approximating an average absorption coefficient

UNFINISHED
"""

import caffe
import sys
import csv
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define absorption coefficients for each material

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

abs_coeff_file = "../abs_coefficients.csv"

def plot_abs_map(network_output, save=False, path=None, band=1000):
    """
    Function for plotting absporption coefficient maps
    :param network_output:
    :param save:
    :param path:
    :param band: frequency band in Hz
    :return:
    """

    if type(network_output) is dict:  # If the input value is an unmodified 'network output'
        class_map = network_output['prob'][0].argmax(axis=0)  # Get highest probability class at each location
    else:  # if average probability maps are passed in in case of upsampling & averaging
        class_map = network_output.argmax(axis=0)

    pixels = np.array([[colormap[class_num] for class_num in row] for row in class_map], dtype=np.uint8)

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.set_title("Absorption Coefficients at " + str(band) + " Hz")
    hb = ax.imshow(pixels)

    if save is True:  # If user chooses to save plot, do so
        if path is None:  # If no path is specified, create one for storing probability maps
            path = os.path.join(os.getcwd(), "plots", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(path)

        plt.savefig(path + "/abs_map.jpg")

    else:  # Just show the plot
        plt.show()


def get_pixel_map(class_map, colormap):
    """
    Function to generate a pixel map (from network output) which can be plotted by OpenCV
    :param class_map:
    :param colormap:
    :return: an array of tuples to be plotted by OpenCV. The tuples define pixel values
    """
    """
    Function to generate a pixel map (from network output) which can be plotted by OpenCV
    :param network_output: output from network (inference on GPU or NCS)
    :return: an array of tuples to be plotted by OpenCV. The tuples define pixel values
    """

    # Convert to format suitable for plotting with OpenCV, i.e. array of pixels
    pixel_map = np.array([[colormap[class_num] for class_num in row] for row in class_map], dtype=np.uint8)
    return pixel_map


def get_coefficients(band, file_path=abs_coeff_file):
    """
    Function returning a dict of materials and corresponding absorption coefficient lists
    :param: band: frequency band to get absorption coefficients from [125,250,500,1000,2000,4000]
    :param file_path: path to file containing absorption coefficients
    :return:
    """

    with open(file_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        headers = next(read_csv, None)  # Get headers in csv file

        if str(band) not in headers:
            print("Invalid frequency band")
            return
        else:
            index = headers.index(str(band))

        print("headers: ")
        print(headers)

        abs_coeffs = []
        for row in read_csv:
            abs_coeffs.append(float(row[index]))  # Get abs coeff for each material at given freq band as float

        return abs_coeffs


def get_color_map():
    hues = np.linspace(0, 179, num=23, dtype=int)

    hsv_colors = []

    for hue in hues:
        hsv_colors.append((hue, 255, 255))

    rgb_colors = []

    for hsv_color in hsv_colors:
        rgb_colors.append(cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2RGB))

    for color in rgb_colors:
        print(color[0][0])


def get_grayscale(c):
    """
    Returns a map of scalar values based on given absorption coefficients.
    Output can be used as greyscale colormap for plotting segmentation results
    :param c: absorption coefficients
    :return:
    """
    min = min(c)
    max = max(c)

    greyscale_values = np.interp(c, [min, max], [0, 255]).astype(int)


if __name__ == "__main__":
    coeffs = get_coefficients(1000)
    get_color_map()







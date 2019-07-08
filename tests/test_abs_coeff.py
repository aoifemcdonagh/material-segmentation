"""
Script for approximating an average absorption coefficient

UNFINISHED
"""

import caffe
import sys
import csv
import numpy as np
import cv2

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





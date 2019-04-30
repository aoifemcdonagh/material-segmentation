"""
Script for approximating an average absorption coefficient
"""

import caffe
import sys

import minc_classify as minc_utils
import segment as resize
import numpy as np


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

abs_coeffs = "/home/gwy-dnn/Documents/Aoife/Reverb/absorption_coeffs.ods"

def get_coefficients(file_path):
    """
    Function returning a dict of materials and corresponding absorption coefficient lists
    :param file_path: path to file containing absorption coefficients
    :return:
    """



def get_average_coeff():
    """
    Function for generating an average absorption coeff based on material classification probabilities
    :return:
    """


if __name__ == "__main__":
    caffe.set_mode_gpu()
    image_path = sys.argv[1]  # path to image to be segmented

    output = minc_utils.classify(image_path)
    class_map = minc_utils.get_class_map(output)
    print(class_map)
    #get_average_coeff()

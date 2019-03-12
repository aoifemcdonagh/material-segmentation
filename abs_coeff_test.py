"""
Script for approximating an average absorption coefficient
"""

import sys
import os
import skimage

import full_image_classify as minc_utils
import classify_resized as resize
from scipy import misc
import numpy as np
from datetime import datetime

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

def get_average_coeff():
    """
    Function for generating an average absorption coeff based on material classification probabilities
    :return:
    """


if __name__ == "__main__":
    image_path = sys.argv[1]  # path to image to be segmented

    output = minc_utils.classify(image_path)

    get_average_coeff()
    
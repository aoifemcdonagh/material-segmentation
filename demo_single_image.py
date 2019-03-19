"""
Script which segments a single image using a Movidius stick on a Raspberry Pi
Similar to other scripts in this directory, but modified for better performance on Movidius stick and Raspberry Pi
"""

import caffe
import sys
import skimage

import full_image_classify as minc_utils
import classify_resized as resize
import numpy as np



if __name__ == "__main__":


import caffe
import sys
import os
import skimage

import full_image_classify as minc_utils
import classify_resized as resize
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


def add_padding(im):
    """
    Function to add padding to an image
    :param im:
    :return:
    """

    return np.pad(im, pad_width=((50, 50), (100, 100), (0, 0)), mode='symmetric')


def plot(im):
    """
    Plot an image
    :param im:
    :return:
    """

    fig, ax = plt.subplots()
    hb = ax.imshow(im)
    ax.set_title("Padded Image")
    plt.show()


if __name__ == "__main__":
    caffe.set_mode_gpu()
    image_path = sys.argv[1]  # path to image to be segmented

    image = caffe.io.load_image(image_path)

    plot(add_padding(image))

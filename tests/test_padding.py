import caffe
import sys
import matplotlib.pyplot as plt
import numpy as np


def add_padding(im, pad):
    """
    Function to add padding to an image
    :param im:
    :param pad: number of pixels to pad around image
    :return:
    """

    return np.pad(im, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='symmetric')


def remove_padding(im, pad):
    """
    Function for removing padding from an image.
    :param im: image to remove padding from
    :param pad: number of pixels of padding to remove
    :return:
    """

    return im[pad:-pad, pad:-pad]


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
    padding = int(sys.argv[2])  # number of pixels to pad

    image = caffe.io.load_image(image_path)
    padded_image = add_padding(image, padding)  # Test adding padding
    plot(padded_image)

    padding_removed = remove_padding(padded_image, padding)  # Test if padding removal works
    plot(padding_removed)

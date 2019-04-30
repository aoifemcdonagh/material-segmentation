import skimage.transform
import numpy as np
import logging as log
import time
import sys


#  List of tuples defining colours for each class.
classes_color_map = [
    (150, 150, 150),
    (58, 55, 169),
    (211, 51, 17),
    (157, 80, 44),
    (23, 95, 189),
    (210, 133, 34),
    (76, 226, 202),
    (101, 138, 127),
    (223, 91, 182),
    (80, 128, 113),
    (235, 155, 55),
    (44, 151, 243),
    (159, 80, 170),
    (239, 208, 44),
    (128, 50, 51),
    (82, 141, 193),
    (9, 107, 10),
    (223, 90, 142),
    (50, 248, 83),
    (178, 101, 130),
    (71, 30, 204),
    (30, 30, 30),
    (90, 90, 90)
]

# List of HSV color pixel values for each class
hsv_color_map = [(0, 255, 255),
 (8, 255, 255),
 (16, 255, 255),
 (24, 255, 255),
 (32, 255, 255),
 (40, 255, 255),
 (48, 255, 255),
 (56, 255, 255),
 (65, 255, 255),
 (73, 255, 255),
 (81, 255, 255),
 (89, 255, 255),
 (97, 255, 255),
 (105, 255, 255),
 (113, 255, 255),
 (122, 255, 255),
 (130, 255, 255),
 (138, 255, 255),
 (146, 255, 255),
 (154, 255, 255),
 (162, 255, 255),
 (170, 255, 255),
 (179, 255, 255)]

# RGB color map with 23 evenly spaced colors (Evenly spaced in HSV spectrum and converted)
even_color_map = [
    (255, 0, 0),
    (255, 68, 0),
    (255, 136, 0),
    (255, 204, 0),
    (238, 255, 0),
    (170, 255, 0),
    (102, 255, 0),
    (34, 255, 0),
    (0, 255,  43),
    (0, 255, 111),
    (0, 255, 179),
    (0, 255, 247),
    (0, 195, 255),
    (0, 127, 255),
    (0, 59, 255),
    (17, 0, 255),
    (85, 0, 255),
    (153, 0, 255),
    (221, 0, 255),
    (255, 0, 221),
    (255, 0, 153),
    (255, 0, 85),
    (255, 0, 8),

]

# TODO: implement color map based on material absorption coeff values

abs_color_map = []

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


def class_to_pixel(c, t='even'):
    """
    Function to convert int values to tuples
    :param c: 2D array of int values between 0 and 22 (material segmentation output)
    :param t: type of color map to use
        even: evenly spaced RGB colors (evenly spaced hue values in HSV space and converted)
        abs: color map based on absorption coeffs for each material
    :return: np array suitable for plotting with cv2. Each class represented by a seperate colour.
    """

    color_map = even_color_map if t == 'even' else abs_color_map  # Use specified color map
    return np.array([[color_map[class_num] for class_num in row] for row in c], dtype=np.uint8)


def get_class_map(network_output, map_type='even'):
    """
    Function to generate a class map which can be plotted by OpenCV
    :param network_output:
    :param map_type: type of color map to use ('even' or 'abs')
    :return: an array of tuples to be plotted by OpenCV. The tuples define pixel values
    """
    if type(network_output) is dict:  # If the input value is an unmodified 'network output' (from a single image)
        class_map = network_output['prob'][0].argmax(axis=0)  # Get highest probability class at each location
    else:  # if average probability maps are passed in in case of upsampling & averaging
        class_map = network_output.argmax(axis=0)  # Get highest probability class at each location

    pixel_map = class_to_pixel(class_map, map_type)  # Convert to format suitable for plotting with OpenCV, i.e. array of pixels

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

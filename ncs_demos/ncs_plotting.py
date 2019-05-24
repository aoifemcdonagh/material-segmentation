# UNFINISHED, not referenced by any other files
# Script containing functions for plotting demo results
# Intended to have everything plotted in OpenCV
# For developing the GUI, it may be easier to have plotting occur in the GUI script?

#import cv2
import numpy as np

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
    (30, 30, 30)
]


def get_class_names(class_numbers):
    """
    Function generating the corresponding class name for a class number outputted by network
    :param class_numbers:
    :return:
    """

    class_names = []
    for number in class_numbers:
        class_names.append(CLASS_LIST.get(number))

    print(class_names)

    if len(class_names) == 1:  # If only one number is passed to function, we should return a string value
        return class_names[0]
    else:  # Otherwise return a list of strings
        return class_names


def get_tick_labels(class_numbers):
    """
    Function generating tick labels appropriate to each classified image
    :param class_numbers:
    :return:
    """

    class_names = get_class_names(class_numbers)
    tick_labels = []
    for (number, name) in zip(class_numbers, class_names):
        tick_labels.append(str(number) + ": " + name)

    return tick_labels


def modify_class_map(class_map):
    """
    Function converting a matrix of numbers (corresponding to discrete, numerically unrelated class values) to a
    matrix of numbers which can be plotted.
    Need to assign numerical values of 0-len(unique_classes) in order to properly plot a class map
        - if len(unique_classes) = 5 for example, instead of a prob_map containing a mix of numbers [0,4,16,19,21]
          we would plot a modified_prob_map containing ndarray of numbers [0,1,2,3,4]
        - Important to use the original unique_classes when generating tick labels! (in this case number corresponds
          to a fixed class name)
    :param class_map:
    :return: modified class map suitable for plotting
    """

    unique_values = np.unique(class_map).tolist()  # list of unique values in class_map
    modified_values = range(0, len(unique_values))  # list in range 0 - len(unique_values)
    value_dict = {a: b for (a, b) in zip(unique_values, modified_values)}

    modified_class_map = [[value_dict.get(class_map[i][j]) for j in range(0, class_map.shape[1])] for i in
                          range(0, class_map.shape[0])]

    return modified_class_map



def get_class_map(network_output):
    """
    Function to generate a class map which can be plotted by OpenCV
    :param network_output:
    :return: an array of tuples to be plotted by OpenCV. The tuples define pixel values
    """
    if type(network_output) is dict:  # If the input value is an unmodified 'network output' (from a single image)
        class_map = network_output['prob'][0].argmax(axis=0)  # Get highest probability class at each location
    else:  # if average probability maps are passed in in case of upsampling & averaging
        class_map = network_output.argmax(axis=0)  # Get highest probability class at each location




# may not need to implement following plotting functions? It might be better to have plotting occur in main script??
# Or do plotting where GUI is implemented? All that's needed from this script is functions for generating
# class/probability maps to plot

def plot_class_map(results):
    """
    Plot class map based on results (classification output)
    :param results:
    :return:
    """

    if type(results) is dict:  # If the input value is an unmodified 'network output' (from a single image)
        class_map = results['prob'][0].argmax(axis=0)  # Get highest probability class at each location
    else:  # if average probability maps are passed in in case of upsampling & averaging
        class_map = results.argmax(axis=0)  # Get highest probability class at each location

    unique_classes = np.unique(class_map).tolist()  # Get unique classes for plot
    class_map = modify_class_map(class_map)  # Modify class_map for plotting



def plot_confidence_map(results):
    """
        Function for plotting the confidence map from classified image
        :param results:
        :return:
        """

    if type(results) is dict:  # If the input value is an unmodified 'network output'
        confidence_map = results['prob'][0].max(axis=0)  # Get highest probability class at each location
    else:  # if average probability maps are passed in in case of upsampling & averaging
        confidence_map = results.max(axis=0)
#  Script to classify material in a full image.
#  Uses MINC GoogLeNet model modified to be fully convolutional
#  Inputs:
#       - path to image to classify
#  Optional inputs:
#       - path to .prototxt file
#       - path to .caffemodel file
#       - option to plot results or not

import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse
from datetime import datetime


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


def classify(im_path, prototxt="models/deploy-googlenet-conv.prototxt", caffemodel="models/minc-googlenet-conv.caffemodel"):
    """
    Function performing material classification across a whole image of arbitrary size.
    Inputs:
        - im_path: path to image
    Optional inputs:
        - prototxt: name of .prototxt file
        - caffemodel : name of .caffemodel file
    """

    # Load network
    net_full_conv = caffe.Net(prototxt, caffemodel, caffe.TEST)

    im = caffe.io.load_image(im_path)
    print "im shape: " + str(im.shape[0])

    print net_full_conv.blobs['data'].data.shape
    net_full_conv.blobs['data'].reshape(1, 3, im.shape[0], im.shape[1])
    print net_full_conv.blobs['data'].data.shape

    transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([104, 117, 124]))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    # make classification map by forward and print prediction indices at each location
    out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

    return out


def plot_output(network_output, image=None, path=None):
    """
    Function for plotting the output of classification model
        - Class map
        - Confidence map
        - All probability maps

    :param network_output: raw output from classification model
    :param image: optional original image to plot alongside classmap
    :param path: path to save plots to
    :return:
    """

    if path is None:  # If no path is specified, create one for storing probability maps
        path = os.path.join(os.getcwd(), "plots", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(path)
    else:
        path = os.path.join(path, "plots")
        os.makedirs(path)

    class_map = network_output['prob'][0].argmax(axis=0)  # Get highest probability class at each location
    prob_map = network_output['prob'][0].max(axis=0)
    unique_classes = np.unique(class_map).tolist()  # Get unique classes for plot
    class_map = modify_class_map(class_map)  # Modify class_map for plotting

    ## Plot image and class map ##
    fig, axs = plt.subplots(ncols=2, figsize=(30,10))
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
    ax = axs[0]
    hb = ax.imshow(mpimg.imread(image))
    ax.set_title("Input image")

    ax = axs[1]
    ax.set_title("Class at each location")
    hb = ax.imshow(class_map, cmap=plt.get_cmap("gist_rainbow", len(unique_classes)))

    step_length = float(len(unique_classes) - 1) / float(
        len(unique_classes))  # Define the step length between ticks for colorbar.
    loc = np.arange(step_length / 2, len(unique_classes), step_length) if len(unique_classes) > 1 else [
        0.0]  # Shift each tick location so that the label is in the middle
    cb = fig.colorbar(hb, ticks=loc)
    cb.set_ticklabels(get_tick_labels(unique_classes))
    cb.set_label('Class Numbers')

    plt.savefig(path + "/class_map.jpg")
    plt.close()

    ## Plot image and confidence map (i.e. highest prob at each location, irrespective of class) ##
    fig, axs = plt.subplots(ncols=2, figsize=(30,10))
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
    ax = axs[0]
    hb = ax.imshow(mpimg.imread(image))
    ax.set_title("Input image")

    ax = axs[1]
    hb = ax.imshow(prob_map)
    ax.set_title("highest probability at each location (irrespective of class)")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Probability')

    plt.savefig(path + "/confidence_map.jpg")
    plt.close()

    # Plot probability maps for all classes
    for class_num in CLASS_LIST.keys():
        prob_map = network_output['prob'][0][class_num]

        fig, ax = plt.subplots()
        hb = ax.imshow(prob_map, cmap='gray')
        ax.set_title("Probability of class " + CLASS_LIST.get(class_num))
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Probability')

        plt.savefig(path + "/" + str(class_num) + ".jpg")
        plt.close()


def plot_probability_maps(probability_maps, path=None):
    """
    Function for plotting probability maps
    Inputs:
        - probability_maps: probability maps for all classes
        - path: optional path to save plots to. Default is in folder 'plots' in current directory

    This function is called from outside this script
    """

    if path is None:  # If no path is specified, create one for storing probability maps
        path = os.path.join(os.getcwd(), "plots", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(path)
    else:
        path = os.path.join(path, "plots")
        os.makedirs(path)

    class_num = 0  # start at class 0
    for prob_map in probability_maps:

        fig, ax = plt.subplots()
        hb = ax.imshow(prob_map, cmap='gray')
        ax.set_title("Probability of class " + CLASS_LIST.get(class_num))
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Probability')

        plt.savefig(path + "/" + str(class_num) + ".jpg")
        plt.close()
        class_num += 1


def get_probability_maps(network_output):
    """
    Function which returns all probability maps in a network output
    Returns a list of probability maps (numpy arrays)
    """
    return [network_output['prob'][0][class_num] for class_num in CLASS_LIST.keys()]


def get_class_names(class_numbers):
    """
    Function generating the corresponding class name for a class number outputted by network
    """
    class_names = []
    for number in class_numbers:
        class_names.append(CLASS_LIST.get(number))

    print class_names

    if len(class_names) == 1:  # If only one number is passed to function, we should return a string value
        return class_names[0]
    else:  # Otherwise return a list of strings
        return class_names


def get_tick_labels(class_numbers):
    """
    Function generating tick labels appropriate to each classified image
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
    """
    unique_values = np.unique(class_map).tolist()  # list of unique values in class_map
    modified_values = range(0, len(unique_values))  # list in range 0 - len(unique_values)
    value_dict = {a: b for (a, b) in zip(unique_values, modified_values)}

    modified_class_map = [[value_dict.get(class_map[i][j]) for j in range(0, class_map.shape[1])] for i in
                          range(0, class_map.shape[0])]

    print("Modified class map")
    print(modified_class_map)

    return modified_class_map

def get_class_map(network_output):
    """
    function taking network output and returning a map of highest probability classes at each location
    Map of class names
    Used for estimating average absorption coeff?
    :param network_output:
    :return:
    """

    return network_output['prob'][0].argmax(axis=0)  # Get highest probability class at each location


if __name__ == "__main__":
    caffe.set_mode_gpu()
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", type=str, help="path to image to be classified")
    parser.add_argument("-m", "--model", type=str, default="models/", help="path to directory containing .caffemodel and .prototxt files")  # Needed??
    parser.add_argument("--prototxt", type=str, default="models/deploy-googlenet-conv.prototxt", help="path to prototxt file")
    parser.add_argument("--caffemodel", type=str, default="models/minc-googlenet-conv.caffemodel", help="path to caffemodel file")
    parser.add_argument("-p", "--plot", action='store_true', help="to plot results")

    args = parser.parse_args()
    im = args.image
    model_dir = args.model  # Is this needed anymore?
    plot = args.plot

    output = classify(im)

    if plot is True:
        plot_output(output, im)

    #plot_probability_maps(output)

#  Script to classify material in a full image.
#  Uses MINC GoogLeNet model modified to be fully convolutional
#  Inputs:
#       - path to image to classify
#       - path to directory containing .caffemodel and .prototxt files (and categories.txt file)

import caffe
import numpy as np
import matplotlib
matplotlib.use('agg')
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


"""
    Function performing material classification across a whole image of arbitrary size.
    Inputs:
        - im_path: path to image
    Optional inputs:
        - prototxt: name of .prototxt file
        - caffemodel : name of .caffemodel file
"""


def classify(im_path, prototxt="models/deploy-googlenet-conv.prototxt", caffemodel="models/minc-googlenet-conv.caffemodel"):
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


"""
    Function for plotting the output of classification model
    Inputs:
        - raw output from classification model
        - optional original image to plot alongside classmap
"""


def plot_output(network_output, image=None):
    class_map = network_output['prob'][0].argmax(axis=0)  # Get highest probability class at each location
    prob_map = network_output['prob'][0].max(axis=0)
    unique_classes = np.unique(class_map).tolist()  # Get unique classes for plot
    class_map = modify_class_map(class_map)  # Modify class_map for plotting

    # show net input and class labels (discrete numbers)
    fig, axs = plt.subplots(ncols=2, figsize=(20, 30))
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

    fig, axs = plt.subplots(ncols=2, figsize=(20, 30))
    ax = axs[0]
    hb = ax.imshow(mpimg.imread(image))
    ax.set_title("Input image")

    ax = axs[1]
    hb = ax.imshow(prob_map)
    ax.set_title("highest probability at each location (irrespective of class)")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Probability')

    plt.show(block=False)


"""
    Function for plotting the probability maps for all 23 classes of the MINC dataset
    Inputs:
        - network output
        - optional path to image to include in plot
        - optional path to save plots to. Default is in folder 'prob_maps' in current directory
"""


def plot_probability_maps(network_output, image=None, path=None):
    path = os.path.join(os.getcwd(), "plots", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(path)

    class_num = 18
    test = network_output['prob'][0][class_num]

    fig, axs = plt.subplots(ncols=2, figsize=(20, 30))
    ax = axs[0]
    hb = ax.imshow(mpimg.imread(image))
    ax.set_title("Input image")

    ax = axs[1]
    hb = ax.imshow(test, cmap='gray')
    ax.set_title("Probability of class " + CLASS_LIST.get(class_num))
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Probability')

    plt.show()

    fig, ax = plt.subplots()
    hb = ax.imshow(test, cmap='gray')
    plt.savefig(path + "/" + str(class_num) + ".jpg")

"""
    Function generating the corresponding class name for a class number outputted by network
"""


def get_class_names(class_numbers):
    class_names = []
    for number in class_numbers:
        class_names.append(CLASS_LIST.get(number))

    print class_names

    if len(class_names) == 1:  # If only one number is passed to function, we should return a string value
        return class_names[0]
    else:  # Otherwise return a list of strings
        return class_names


"""
    Function generating tick labels appropriate to each classified image
"""


def get_tick_labels(class_numbers):
    class_names = get_class_names(class_numbers)
    tick_labels = []
    for (number, name) in zip(class_numbers, class_names):
        tick_labels.append(str(number) + ": " + name)

    return tick_labels


"""
    Function converting a matrix of numbers (corresponding to discrete, numerically unrelated class values) to a
    matrix of numbers which can be plotted.
    Need to assign numerical values of 0-len(unique_classes) in order to properly plot a class map
        - if len(unique_classes) = 5 for example, instead of a prob_map containing a mix of numbers [0,4,16,19,21]
          we would plot a modified_prob_map containing ndarray of numbers [0,1,2,3,4]
        - Important to use the original unique_classes when generating tick labels! (in this case number corresponds
          to a fixed class name)
"""


def modify_class_map(class_map):
    unique_values = np.unique(class_map).tolist()  # list of unique values in class_map
    modified_values = range(0, len(unique_values))  # list in range 0 - len(unique_values)
    value_dict = {a: b for (a, b) in zip(unique_values, modified_values)}

    modified_class_map = [[value_dict.get(class_map[i][j]) for j in range(0, class_map.shape[1])] for i in
                          range(0, class_map.shape[0])]

    print("Modified class map")
    print(modified_class_map)

    return modified_class_map


if __name__ == "__main__":
    caffe.set_mode_gpu()
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", type=str, help="path to image to be classified")
    parser.add_argument("-m", "--model", type=str, default="models/", help="path to directory containing .caffemodel and .prototxt files")
    parser.add_argument("--prototxt", type=str, default="models/deploy-googlenet-conv.prototxt", help="path to prototxt file")
    parser.add_argument("--caffemodel", type=str, default="models/minc-googlenet-conv.caffemodel", help="path to caffemodel file")
    parser.add_argument("-p", "--plot", type=bool, default=True, help="to plot results")

    args = parser.parse_args()
    im = args.image
    model_dir = args.model
    plot = args.plot

    output = classify(im)

    if plot is True:
        plot_output(output, im)

    plot_probability_maps(output, im)

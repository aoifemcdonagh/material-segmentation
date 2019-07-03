####### REDUNDANT #######
# This script is only used by test_upsampling.py for classify() function.
# Keeping for reference at this point.


import os
os.environ['GLOG_minloglevel'] = '2'  # Supressing caffe printouts of network initialisation
import caffe
import numpy as np

top_dir = os.path.dirname(os.path.realpath(__file__))


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


def classify(im, prototxt=top_dir+"/models/minc-googlenet-conv.prototxt", caffemodel=top_dir+"/models/minc-googlenet-conv.caffemodel"):
    """
    Function performing material classification across a whole image of arbitrary size.

    :param im: image preloaded using caffe.io.load_image()
    :param prototxt: name of .prototxt file
    :param caffemodel: name of .caffemodel file
    :return: network output
    """

    net_full_conv = caffe.Net(prototxt, caffemodel, caffe.TEST)  # Load network
    print(net_full_conv.blobs['data'].data.shape)
    net_full_conv.blobs['data'].reshape(1, 3, im.shape[0], im.shape[1])  # Reshape the input layer to image size
    print(net_full_conv.blobs['data'].data.shape)

    transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([104, 117, 124]))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    # make classification map by forward and print prediction indices at each location
    out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

    return out


def get_probability_maps(network_output):
    """
    Function which returns all probability maps in a network output
    Returns a list of probability maps (numpy arrays)
    """
    return [network_output['prob'][0][class_num] for class_num in CLASS_LIST.keys()]


def get_class_map(network_output):
    """
    function taking network output and returning a map of highest probability classes at each location
    Map of class names
    Used for estimating average absorption coeff?
    :param network_output:
    :return:
    """

    return network_output['prob'][0].argmax(axis=0)  # Get highest probability class at each location




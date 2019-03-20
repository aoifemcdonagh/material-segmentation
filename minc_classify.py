import caffe
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


def classify(im, prototxt="models/deploy-googlenet-conv.prototxt", caffemodel="models/minc-googlenet-conv.caffemodel"):
    """
    Function performing material classification across a whole image of arbitrary size.

    :param im: image preloaded using caffe.io.load_image()
    :param prototxt: name of .prototxt file
    :param caffemodel: name of .caffemodel file
    :return: network output
    """

    # Load network
    net_full_conv = caffe.Net(prototxt, caffemodel, caffe.TEST)

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


if __name__ == "__main__":
    """
    Script to classify material in a full image.
    Uses MINC GoogLeNet model modified to be fully convolutional
    Inputs:
         - path to image to classify
    Optional inputs:
         - path to .prototxt file
         - path to .caffemodel file
         - option to plot results or not
    """
    import argparse
    import minc_plotting as minc_plot

    caffe.set_mode_gpu()
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", type=str, help="path to image to be classified")
    parser.add_argument("-m", "--model", type=str, default="models/", help="path to directory containing .caffemodel and .prototxt files")  # Needed??
    parser.add_argument("--prototxt", type=str, default="models/deploy-googlenet-conv.prototxt", help="path to prototxt file")
    parser.add_argument("--caffemodel", type=str, default="models/minc-googlenet-conv.caffemodel", help="path to caffemodel file")
    parser.add_argument("-p", "--plot", action='store_true', help="to plot results")

    args = parser.parse_args()
    im_path = args.image
    model_dir = args.model  # Is this needed anymore?
    plot = args.plot

    image = caffe.io.load_image(im_path)  # must load image using caffe.io.load_image()
    output = classify(image)

    if plot is True:
        minc_plot.plot_class_map(output)

    #plot_probability_maps(output)

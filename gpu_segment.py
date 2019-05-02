import os
os.environ['GLOG_minloglevel'] = '2'  # Supressing caffe printouts of network initialisation
import caffe
import numpy as np
import skimage

top_dir = os.path.dirname(os.path.realpath(__file__))

SCALES = [1.0 / np.sqrt(2), 1.0, np.sqrt(2)]  # Define scales as per MINC paper

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


def segment(im, pad=0):
    """
    TODO: Function needs to be implemented which performs material classification on an image at three different
    scales. What should be the output based on next stage of processing? Probability maps for a whole image? Separate
    probability maps for each class for each image?
    possibly give method another name which better reflects its function

    This function currently does too much??

    This method should call 'classify()' function from full_image_classify

    :param im: image to segment
    :param pad: number of pixels of padding to add
    :return:
    """

    padded_image = add_padding(im, pad)  # Add padding to original image
    resized_images = resize_images(padded_image)  # Resize original images

    outputs = [classify(image) for image in resized_images]  # Perform classification on images

    average_prob_maps = get_average_prob_maps(outputs, im.shape, pad)

    return average_prob_maps


def get_average_prob_maps(network_outputs, shape, pad=0):
    """
    :param network_outputs: List of outputs
    :param shape: shape of original image needed for upsampling
    :return: Probability maps for each class, averaged from resized images probability maps
    """

    # Get probability maps for each class for each image
    prob_maps = [get_probability_maps(out) for out in network_outputs]

    # Upsample probability maps to dimensions of original image (plus any padding)
    upsampled_prob_maps = upsample(prob_maps, output_shape=(shape[0] + pad*2, shape[1] + pad*2))

    # Probability maps for each class, averaged from resized images probability maps
    averaged_prob_maps = np.average(upsampled_prob_maps, axis=0)

    # Remove the padded sections from the averaged prob maps
    averaged_prob_maps = [remove_padding(prob_map, pad) for prob_map in averaged_prob_maps]

    return np.array(averaged_prob_maps)


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


def classify(im, prototxt=top_dir+"/models/deploy-googlenet-conv.prototxt", caffemodel=top_dir+"/models/minc-googlenet-conv.caffemodel"):
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

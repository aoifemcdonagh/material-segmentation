import caffe
import numpy as np
import matplotlib.pyplot as plt
import sys

caffe.set_mode_gpu()


def classify(path):
    # Load network
    net_full_conv = caffe.Net('../deploy-googlenet-conv.prototxt', '../minc-googlenet-conv.caffemodel', caffe.TEST)

    # load input and configure preprocessing
    im = caffe.io.load_image(path)
    print "im shape: " + str(im.shape[0])

    print net_full_conv.blobs['data'].data.shape
    net_full_conv.blobs['data'].reshape(1, 3, im.shape[0], im.shape[1])
    print net_full_conv.blobs['data'].data.shape

    transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([104,117,124]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    # make classification map by forward and print prediction indices at each location
    out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    print out['prob'][0].argmax(axis=0)

    # show net input and class labels (discrete numbers)
    fig, axs = plt.subplots(ncols=2, figsize=(20,30))
    fig.subplots_adjust(hspace=0.5, left = 0.07, right = 0.93)
    ax = axs[0]
    hb = ax.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
    ax.set_title("Input image")

    ax = axs[1]
    hb = ax.imshow(out['prob'][0].argmax(axis=0), cmap='tab20')
    ax.set_title("Class at each location")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Class Numbers')

    #plt.show()

    # show net input and confidence map (probability of the top prediction at each location)
    fig, axs = plt.subplots(ncols=2, figsize=(20,30))
    fig.subplots_adjust(hspace=0.5, left = 0.07, right = 0.93)
    ax = axs[0]
    hb = ax.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
    ax.set_title("Input image")

    ax = axs[1]
    hb = ax.imshow(out['prob'][0,22], cmap='tab20')
    ax.set_title("Confidence map (probability of the top prediction at each location)")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Confidence')

    #plt.show()

    fig, axs = plt.subplots(ncols=2, figsize=(20,30))
    ax = axs[0]
    hb = ax.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
    ax.set_title("Input image")

    ax = axs[1]
    hb = ax.imshow(out['prob'][0].max(axis=0))
    ax.set_title("highest probability at each location (irrespective of class)")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Probability')

    plt.show()

if __name__ == "__main__":
    im_path = sys.argv[1]
    classify(im_path)

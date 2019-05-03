# Script for testing fully conv network on image of arbitrary size

import caffe
import numpy as np
import matplotlib.pyplot as plt
import sys

caffe.set_mode_gpu()

# Load the fully convolutional network
net=caffe.Net('minc-googlenet-conv.prototxt','minc-googlenet-conv.caffemodel', caffe.TEST)

print(net.inputs)
print(net.blobs['data'].data.shape)

im = caffe.io.load_image(sys.argv[1])
net.blobs['data'].reshape(10,3,im.shape[0],im.shape[1])  # Reshape input layer to accept image of arbitrary dimension
print(net.blobs['data'].data.shape)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.array([104,117,124]))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

transformed_image = transformer.preprocess('data', im)

out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
print(out['prob'][0].argmax(axis=0))

plt.imshow(im)
plt.show()

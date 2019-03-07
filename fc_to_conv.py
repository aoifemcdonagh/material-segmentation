# Script for converting a GoogLeNet model to be fully convolutional.
# Fully connected layers in the model are cast as convolutional layers.

# Assumes models are in the current working directory. Change model names/paths as necessary.

import caffe
caffe.set_mode_gpu()

net = caffe.Net('deploy-googlenet.prototxt', 'minc-googlenet.caffemodel', caffe.TEST)
params = ['fc8-20']

# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in fc_params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

net_full_conv = caffe.Net('deploy-googlenet-conv.prototxt', 'minc-googlenet.caffemodel', caffe.TEST)
params_full_conv = ['fc8-conv']

conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

# Flatten the weights of the fc layer
for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save('minc-googlenet-conv.caffemodel')

for param in conv_params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(param, conv_params[param][0].shape, conv_params[param][1].shape)

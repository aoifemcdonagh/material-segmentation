# Taking model pretrained on MINC dataset and casting as fully connected CNN
# This is one step towards performing image segmentation based on material recognition as described in MINC paper
# based on net_surgery caffe example

import caffe
caffe.set_mode_gpu()

net = caffe.Net('deploy-googlenet.prototxt', 'minc-googlenet.caffemodel', caffe.TEST)

params = ['fc8-20']
# fc_params = {name: (weights, biases)}

fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('deploy-googlenet-conv.prototxt',
                          'minc-googlenet.caffemodel',
                          caffe.TEST)
params_full_conv = ['fc8-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

# Transplanting parameters?
for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save('minc-googlenet-conv.caffemodel') # Saving the new model weights

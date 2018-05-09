#!/usr/bin/python
# Generate the network configuration

import os
import caffe
from caffe import layers as L, params as P

# Default convolutional layer
def Convolution(bottom, num_output=64, kernel_size=3, stride=1, pad=1,
                weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
                param=[dict(lr_mult=1),dict(lr_mult=0.1)]):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         weight_filler=weight_filler, bias_filler=bias_filler, param=param)
    return conv

# Default deconvolutional layer
def Deconvolution(bottom, num_output=1, kernel_size=[19,1,1], stride=[5,1,1], pad=[7,0,0], param=[dict(lr_mult=0.1, decay_mult=0), dict(lr_mult=0, decay_mult=0)]):

    deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=num_output,kernel_size=kernel_size,
                                                            stride=stride,pad=pad,weight_filler=dict(type='gaussian',std=0.001),
                                                            bias_filler=dict(type='constant', value=0)),param=param)
    return deconv

# Default convolutional and prelu layer
def Convolution_PReLU(bottom, num_output=64, kernel_size=3, stride=1, pad=1,
                      weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
                      param=[dict(lr_mult=1),dict(lr_mult=0.1)]):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         weight_filler=weight_filler, bias_filler=bias_filler, param=param)
    relu = L.PReLU(conv, in_place=True)
    return conv, relu

# Default convolutional, bn and prelu bn layer
def Convolution_BN_PReLU(bottom, num_output=64, kernel_size=3, stride=1, pad=1,
                         weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
                         param=[dict(lr_mult=1),dict(lr_mult=0.1)]):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         weight_filler=weight_filler, bias_filler=bias_filler, param=param)
    bn   = L.BatchNorm(conv, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
    relu = L.PReLU(bn, in_place=True)
    return conv, bn, relu

# Default convolutional, bn and relu bn layer
def Convolution_BN_ReLU(bottom, num_output=64, kernel_size=3, stride=1, pad=1,
                         weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
                         param=[dict(lr_mult=1),dict(lr_mult=0.1)]):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         weight_filler=weight_filler, bias_filler=bias_filler, param=param)
    bn   = L.BatchNorm(conv, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
    relu = L.ReLU(bn, in_place=True)
    return conv, bn, relu

def gen_net(train_hdf5_in, train_batch_size, test_hdf5_in, test_batch_size, deploy=False):

    # Input Layers
    n = caffe.NetSpec()
    if deploy:
        n.data              = L.DummyData(ntop=1, shape=[dict(dim=[1, 1, 20, 20, 20])])
    else:
        n.data,  n.label    = L.HDF5Data(ntop=2, include=dict(phase=caffe.TRAIN), hdf5_data_param=dict(batch_size=train_batch_size), source=train_hdf5_in )
        n.data2             = L.HDF5Data(ntop=0, top=['data','label'], include=dict(phase=caffe.TEST),  hdf5_data_param=dict(batch_size=test_batch_size), source=test_hdf5_in)

    # Core Architecture
    n.deconv1               = Deconvolution(n.data)
    n.conv1, n.bn1, n.relu1 = Convolution_BN_ReLU(n.deconv1,num_output=64)
    n.conv2, n.bn2, n.relu2 = Convolution_BN_ReLU(n.relu1,num_output=64)
    n.conv3, n.bn3, n.relu3 = Convolution_BN_ReLU(n.relu2,num_output=32)
    n.conv4, n.bn4, n.relu4 = Convolution_BN_ReLU(n.relu3,num_output=16)
    n.conv5, n.bn5, n.relu5 = Convolution_BN_ReLU(n.relu4,num_output=16)
    n.conv6                 = Convolution        (n.relu5,num_output=1, param=[dict(lr_mult=0.1),dict(lr_mult=0.1)])
    n.recon                 = L.Eltwise(n.deconv1, n.conv6, operation=P.Eltwise.SUM)

    # Output Layers
    if not deploy:
        n.loss  = L.EuclideanLoss(n.recon, n.label)
        #n.loss = L.Python (n.recon, n.label, python_param=dict(module='pyloss',layer='SmoothL1LossLayer_2'),loss_weight=1)

    # Return the network
    return n.to_proto()

if __name__ == '__main__':

    dest_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dest_dir, "training/SRCNN_net_cmr.prototxt"), 'w') as f:
        f.write(str(gen_net("{0}/training/train_cmr.txt".format(dest_dir), 64, "{0}/training/test_cmr.txt".format(dest_dir),10, False)))

    with open(os.path.join(dest_dir, "inference/SRCNN_deploy_cmr.prototxt"), 'w') as f:
        f.write(str(gen_net("{0}/training/train_cmr.txt".format(dest_dir), 64, "{0}/training/test_cmr.txt".format(dest_dir),10, True)))
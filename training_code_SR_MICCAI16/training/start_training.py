__author__ = 'oo2113'


import numpy as np
import os
import sys
sys.path.insert(1,'/vol/biomedic/users/oo2113/lib/caffe-alimia/python')
import caffe


# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# Cubic filter
def cubic(x):
    # Reference: Robert G. Keys, "Cubic Convolution Interpolation for Digital Image Processing",
    # IEEE Transactions on Acoustics, Speech, and Signal Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.
    # Our implementation is Equation (15) in the paper and Lines 862-875 in the Matlab function imresize.m.
    s  = np.abs(x)
    s2 = s * s
    s3 = s2 * s
    f  = (1.5*s3 - 2.5*s2 + 1) * (s <= 1) + (-0.5*s3 + 2.5*s2 - 4*s + 2) * ((1 < s) & (s <= 2))
    return f

def initialize_net_with_cubic(net):

    # Upsampling factor
    local_scale_z = 5.0

    # Modify the deconvolution weights
    # The length of x is the same as the kernel_size
    local_x = np.arange(-2+1.0/local_scale_z, 2, 1.0/local_scale_z)
    net.params['deconv1'][0].data[0,0,:,0,0] = cubic(local_x)
    net.params['deconv1'][1].data[:] = 0

# Main Training
# base net -- follow the editing model parameters example to make
dest_dir    = os.path.dirname(os.path.realpath(__file__))
solver_name = os.path.join(dest_dir,'SRCNN_solver_cmr.prototxt')

caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.AdamSolver(solver_name)

# Initialize the upsampling with bicubic kernel
#initialize_net_with_cubic(solver.net)
# Modify the deconvolution weights
# The length of x is the same as the kernel_size
scale_z = 5
x = np.arange(-2+1.0/scale_z, 2, 1.0/scale_z)
solver.net.params['deconv1'][0].data[0,0,:,0,0] = cubic(x)
solver.net.params['deconv1'][1].data[:] = 0

# Print the filter coefficients
print 'trainnet deconv: ', solver.net.params['deconv1'][0].data

# solve straight through -- a better approach is to define a solving loop to
solver.solve()
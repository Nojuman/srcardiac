#!/usr/bin/python
# Generate the solver protocol

with open('SRCNN_solver_cmr.prototxt', 'w') as f:

    f.write('# The train/test net protocol buffer definition\n')
    f.write('net: "SRCNN_net_cmr.prototxt"\n\n')
    
    f.write('# test_iter specifies how many forward passes the test should carry out. In our case,\n')
    f.write('# we have test batch size 10 and 556 test iterations, covering almost the full 5560 patches.\n')
    f.write('test_iter: 556\n\n')
    f.write('# Carry out testing every 500 training iterations.\n')
    f.write('test_interval: 500\n\n')
    
    f.write('# The base learning rate, momentum and the weight decay of the network.\n')
    f.write('type: "Adam"\n')
    f.write('base_lr: 0.0001\n')
    f.write('lr_policy: "fixed"\n')
    f.write('momentum: 0.9\n')
    f.write('momentum2: 0.995\n')
    f.write('weight_decay: 0.0001\n\n')

    f.write('# The maximum number of iterations\n')
    f.write('# For 380000 training samples and a batch size of 256, one epoch is about 1484 iterations.\n')
    f.write('# So 300000 iterations make about 200 epochs.\n')
    f.write('max_iter: 300000\n\n')
    
    f.write('# Display every 500 iterations\n')
    f.write('display: 500\n\n')
    f.write('# snapshot intermediate results\n')
    f.write('snapshot: 5000\n')
    f.write('snapshot_prefix: "tmp/SRCNN_cmr"\n\n')
    
    f.write('# solver mode: CPU or GPU\n')
    f.write('solver_mode: GPU\n')
    
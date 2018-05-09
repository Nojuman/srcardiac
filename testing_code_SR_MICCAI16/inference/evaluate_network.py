__author__ = 'oo2113'
from deploy_net import deploy_network_3D
from deploy_net import upsample_withothermethods
import os
import time

def mkdirfun(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Methods to evaluate
methods         = ['cnn','bspline','linear']
num_evaluations = 3

# Subject Selection
currentwd       = os.getcwd()
inputimagedir   = os.path.join(currentwd,'../example_images/input_2D');  inputimagenames  = sorted(os.walk(inputimagedir).next()[2]); inputimagenames  = [inputimagedir+'/'+inputimagename for inputimagename in inputimagenames]
inputimagenames = inputimagenames[:num_evaluations]

# Model Definition
networkfile = os.path.join(currentwd, 'SRCNN_deploy_cmr.prototxt')
modelfile   = os.path.join(currentwd, 'SRCNN_cmr_upsample_x5.caffemodel')
#modelfile   = os.path.join(currentwd, 'SRCNN_cmr_iter_75000.caffemodel')

# imagename for loop is parallelizable
for loopId,inputimagename in enumerate(inputimagenames):

    # Flags for linear and bspline interpolation
    print inputimagename
    print 'image {0} out of {1}'.format(loopId+1,num_evaluations)

    # Parameter Initialization
    outputdirs  = {method: os.path.join(currentwd,'../example_images/output_'+method) for method in methods}
    supimgnames = {method: outputdirs[method]+'/'+inputimagename.split('/')[-1] for method in methods}
    for method in methods: mkdirfun(outputdirs[method])

    # Compute the computation time
    start_time = time.time()

    # Generate the high resolution images
    if 'cnn' in methods:
        deploy_network_3D(networkfile,modelfile,inputimagename,supimgnames['cnn'])

    if 'bspline' in methods:
       upsample_withothermethods(inputimagename,supimgnames['bspline'],5.0,'bspline')

    if 'linear' in methods:
        upsample_withothermethods(inputimagename,supimgnames['linear'],5.0,'linear')

    # Report the execution time
    print("--- %s seconds ---" % (time.time() - start_time))
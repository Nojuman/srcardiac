def is_even(num):
    """Return whether the number num is odd."""
    return num % 2 == 0

def is_odd(num):
    """Return whether the number num is odd."""
    return num % 2 == 1

def learn_spatial_support(networkname,modelname):

    # Import libraries
    import numpy as np
    import caffe; caffe.set_mode_cpu()

    # Read the model
    mynetwork = caffe.Net(networkname,caffe.TEST)
    mynetwork.copy_from(modelname)

    # Spatial Support
    spatial_support = np.array([0,0,0])
    for layer_name, layer in mynetwork.params.items():
        if layer_name.startswith('conv'):
            x_sup = layer[0].data.shape[4]-1
            y_sup = layer[0].data.shape[3]-1
            z_sup = layer[0].data.shape[2]-1
            spatial_support += np.array([x_sup,y_sup,z_sup])
    return spatial_support

def readimage_preprocess_deconv(inputimagename,upscale_factor):

    import SimpleITK as sitk
    import numpy as np

    inputimage  = sitk.ReadImage(inputimagename)
    inputarray  = np.swapaxes(sitk.GetArrayFromImage(inputimage),0,2).astype(float)
    inputsize   = inputarray.shape
    outputsize  = [inputsize[0],inputsize[1],inputsize[2]*upscale_factor]
    maxval      = np.max(inputarray[:])
    inputarray /= maxval

    return inputarray, maxval, outputsize

def run_3D_caffe_model (inputarray,networkname,modelname,spat_sup,outputsize):

    # Import libraries
    import caffe; caffe.set_mode_gpu()
    import numpy as np

    # Size of the input array
    inputsize    = inputarray.shape
    output_array = np.zeros(outputsize)

    # Parse the data into N chunks
    batchsize_x = 11
    if is_even(batchsize_x ) | is_odd(spat_sup[0]):
        raise ValueError('3D conv patch size is not an odd number')  # ensure that it is an odd number;
    radiussize_x  = (batchsize_x-1)/2.0
    radius_spat_x = spat_sup[0]/2.0
    center_pnts_x = np.arange(radiussize_x,inputsize[0]+radiussize_x,batchsize_x)

    # Loop over all the batches
    for center_pnt_x in center_pnts_x:
        x_b_low  = (center_pnt_x - radiussize_x) - radius_spat_x
        x_b_high = (center_pnt_x + radiussize_x) + radius_spat_x
        if x_b_low<0:                 x_b_low=0.0
        if x_b_high>(inputsize[0]-1): x_b_high=(inputsize[0]-1)
        patcharray = inputarray[x_b_low:x_b_high+1,:,:]

        # Read the model
        mynetwork = caffe.Net(networkname, caffe.TEST)
        mynetwork.copy_from(modelname)

        # Update the network input array
        network_input  = np.expand_dims(np.swapaxes(patcharray,0,2),axis=0)
        mynetwork.blobs['data'].reshape(1, *network_input.shape)
        mynetwork.blobs['data'].data[...] = network_input

        # perform inference
        mynetwork.forward()
        output_layer_name = mynetwork.outputs[0]
        batch_array       = mynetwork.blobs[output_layer_name].data # be careful z and x are swapped

        # build the output image from patches
        x_i_low  = (center_pnt_x - radiussize_x)
        x_i_high = (center_pnt_x + radiussize_x)
        if x_i_low<0:                 x_i_low=0.0
        if x_i_high>(inputsize[0]-1): x_i_high=(inputsize[0]-1)
        b_start = x_i_low-x_b_low
        b_end   = b_start + (x_i_high-x_i_low)
        
        output_array[x_i_low:x_i_high+1,:,:] = np.squeeze(np.swapaxes(batch_array[:,:,:,:,b_start:b_end+1],2,4))

    return output_array

def deploy_network_3D(networkfile,modelfile,inputimagenames,outputimagename):

    # Load the libraries
    import SimpleITK as sitk
    import numpy as np

    # parameters
    upscale_factor = 5.0

    # read the image and preprocess it
    inputarray, maxval, outputsize = readimage_preprocess_deconv (inputimagenames,upscale_factor)
    inputsize = inputarray.shape

    # Initial Read of the model - to learn the spatial support
    spatial_support=learn_spatial_support(networkfile,modelfile)

    #load the network and create batches from the input image
    output_array = run_3D_caffe_model (inputarray,networkfile,modelfile,spatial_support,outputsize)

    # Mask the negative valued pixels
    output_array *= maxval
    output_array[output_array < 0.0] = 0.0
    output_array[output_array > maxval] = maxval

    # Set new image spacing and image origin
    inputimage_cont = sitk.ReadImage(inputimagenames)
    output_image    = sitk.GetImageFromArray(np.swapaxes(output_array,0,2))
    slice_dif       = ((outputsize[2]-1.0)/2.0) - ((inputsize[2]-1.0)*upscale_factor/2.0)
    new_spacing     = tuple(itema/itemb for itema,itemb in zip(inputimage_cont.GetSpacing(),(1,1,upscale_factor)))
    new_origin      = tuple(itema-itemb for itema,itemb in zip(inputimage_cont.GetOrigin() ,(0,0,slice_dif*new_spacing[2])))
    new_direction   = inputimage_cont.GetDirection()

    # Write the Image
    output_image.SetSpacing(new_spacing)
    output_image.SetOrigin(new_origin)
    output_image.SetDirection(new_direction)
    sitk.WriteImage(output_image,outputimagename)

def upsample_withothermethods (inputname,outputname,upscale_factor,methodname):
    import SimpleITK as sitk

    # read the image and find the new resolution
    inputimage       = sitk.ReadImage(inputname)
    original_spacing = inputimage.GetSpacing()
    original_size    = inputimage.GetSize()
    new_spacing      = [original_spacing[0],original_spacing[1],original_spacing[2]/upscale_factor]
    new_size         = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))]
    resampleSliceFilter = sitk.ResampleImageFilter()

    # upscale the low resolution image with the given method
    if methodname == 'bspline':
        outputimage = resampleSliceFilter.Execute(inputimage, new_size, sitk.Transform(),
                                                  sitk.sitkBSpline, inputimage.GetOrigin(),
                                                  new_spacing, inputimage.GetDirection(), 0,
                                                  inputimage.GetPixelIDValue())
        sitk.WriteImage(outputimage, outputname)

    if methodname == 'linear':
        outputimage = resampleSliceFilter.Execute(inputimage, new_size, sitk.Transform(),
                                                  sitk.sitkLinear, inputimage.GetOrigin(),
                                                  new_spacing, inputimage.GetDirection(), 0,
                                                  inputimage.GetPixelIDValue())
        sitk.WriteImage(outputimage, outputname)







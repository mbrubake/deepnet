#!/usr/bin/python
import time
import numpy as np
import os
from scipy import ndimage as spnd
from scipy.ndimage import morphology as morph
from scipy import misc
from glob import glob
from optparse import OptionParser
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cmdLineParser = OptionParser()
    cmdLineParser.add_option('--patch_size',type='int',dest='patch_size',default=64)
    cmdLineParser.add_option('--patch_stride',type='int',dest='patch_stride',default=8)
    cmdLineParser.add_option('--im_start',type='int',dest='im_start',default=0)

    (options, args) = cmdLineParser.parse_args()

    indir = args[0];
    outdir = args[1];
    sat_filelist = sorted(glob(os.path.join(indir,'sat','*.tiff')))
    map_filelist = sorted(glob(os.path.join(indir,'map','*.tif')))
    num_road_pixels = 0
    num_pixels = 0
    for (file_ind,(satimFile,mapimFile)) in enumerate(zip(sat_filelist,map_filelist)):
        assert os.path.splitext(os.path.split(satimFile)[1])[0] == os.path.splitext(os.path.split(mapimFile)[1])[0]
        print '\nImage: {0}'.format(os.path.split(satimFile)[1])
        satim = misc.imread(satimFile)
        mapim = misc.imread(mapimFile)

        left,right,top,bottom = 0,satim.shape[1],0,satim.shape[0]
	while np.all(satim[:,left,:] > 253): left+=1
        while np.all(satim[:,right-1,:] > 253): right-=1
        while np.all(satim[top,:,:] > 253): top+=1
        while np.all(satim[bottom-1,:,:] > 253): bottom-=1

        print 'Cropping to {0},{1} x {2}, {3}'.format(left,top,right,bottom)
        outsatim = satim[top:bottom,left:right,:]
        outmapim = mapim[top:bottom,left:right]

        t = time.time()
        outvalim = np.logical_not(morph.binary_dilation(morph.binary_erosion(np.all(outsatim > 253,axis=2),iterations=8),iterations=16))
        print 'Took {0}s to detect valid pixels'.format(time.time() - t)

        outsatimFile = os.path.join(outdir,'sat',os.path.splitext(os.path.split(satimFile)[1])[0] + '.png')
        outmapimFile = os.path.join(outdir,'map',os.path.splitext(os.path.split(mapimFile)[1])[0] + '.png')
        outvalimFile = os.path.join(outdir,'val',os.path.splitext(os.path.split(mapimFile)[1])[0] + '.png')

        misc.imsave(outsatimFile,outsatim)
        misc.imsave(outmapimFile,outmapim)
        misc.imsave(outvalimFile,255*outvalim.astype('uint8'))

        num_pixels += np.sum(outvalim)
        num_road_pixels += np.sum(outmapim[outvalim])
        print '{0} out of {1} road pixels = {2}%'.format(num_road_pixels,num_pixels, float(num_road_pixels)/float(num_pixels))

#if __name__ == '__main__':
if 0:
    cmdLineParser = OptionParser()
    cmdLineParser.add_option('--patch_size',type='int',dest='patch_size',default=64)
    cmdLineParser.add_option('--patch_stride',type='int',dest='patch_stride',default=8)
    cmdLineParser.add_option('--im_start',type='int',dest='im_start',default=0)

    (options, args) = cmdLineParser.parse_args()

    indir = args[0];
    outfile_base = args[1];
    sat_filelist = sorted(glob('{0}/sat/*.tiff'.format(indir)))
    map_filelist = sorted(glob('{0}/map/*.tif'.format(indir)))
    for (file_ind,(satimFile,mapimFile)) in enumerate(zip(sat_filelist,map_filelist)):
        print 'Satellite: {0}\nMap: {1}'.format(satimFile,mapimFile);
        t = time.time()
        satim = misc.imread(satimFile).astype(np.float32) / 255.0
        print 'Time spent loading image: {0}s'.format(time.time() - t)
#        mapim = misc.imread(mapimFile)

#        num_patches = np.ceil((satim.shape[0] - options.im_start)/options.patch_stride) * np.ceil((satim.shape[1] - options.im_start)/options.patch_stride)
#        imData = np.empty([satim.shape[2]*options.patch_size**2, num_patches],np.uint8)
#        mapData = np.empty([num_patches],np.uint8)
        patchCenters = np.mgrid[options.im_start + int(np.sqrt(2.0)*options.patch_size/2.0) : satim.shape[0]-2*options.patch_size : options.patch_stride,
                                options.im_start + int(np.sqrt(2.0)*options.patch_size/2.0) : satim.shape[1]-2*options.patch_size : options.patch_stride].reshape((2,1,-1))
        patchCenters += 0.5
        nPatches = patchCenters.shape[2]
        nPatchPts = options.patch_size**2

        t = time.time()
        patchPts = np.mgrid[-options.patch_size/2:options.patch_size/2,
                            -options.patch_size/2:options.patch_size/2].reshape((2,nPatchPts))
        print 'Time spent getting patch points: {0}s'.format(time.time() - t)
        
        cangles = 2.0*np.pi*np.random.rand(nPatches)
        t = time.time()
        rots = np.array([[ np.cos(cangles), np.sin(cangles)],
                         [-np.sin(cangles), np.cos(cangles)]],dtype=np.float64).transpose((2,0,1))
        print 'Time spent constructing rotations: {0}s'.format(time.time() - t)
        
        t = time.time()
        patchCoords = np.dot(rots, \
                             patchPts.reshape((1,2,nPatchPts))).reshape((nPatches,2,nPatchPts)) \
                                                               .transpose((1,2,0)) \
                      + patchCenters.reshape((2,1,nPatches))
        print 'Time spent computing patch coordinates: {0}s'.format(time.time() - t)

        t = time.time()
        patches = np.empty((satim.shape[2],nPatchPts,nPatches),dtype=np.float32)
        for cdim in range(0,satim.shape[2]):
            spnd.interpolation.map_coordinates(satim[:,:,cdim],patchCoords,output=patches[cdim,:,:],order=0);
        print 'Time spent interpolating: {0}s'.format(time.time() - t)

        for i in range(0,nPatches):
            cpatch = patches[:,:,i].transpose().reshape((options.patch_size,options.patch_size,-1))
            x = patchCenters[0,0,i]
            y = patchCenters[1,0,i]
            satPatch = satim[x - options.patch_size/2 : x + options.patch_size/2,
                             y - options.patch_size/2 : y + options.patch_size/2]
            plt.figure(1)
            plt.imshow(satPatch)
            plt.figure(2)
            plt.imshow(cpatch)
            plt.show()

        i = 0;
        for x in range(options.im_start + int(np.sqrt(2.0)*options.patch_size/2.0),satim.shape[0]-2*options.patch_size,options.patch_stride):
            for y in range(options.im_start + int(np.sqrt(2.0)*options.patch_size/2.0),satim.shape[1]-2*options.patch_size,options.patch_stride):
                satPatch = satim[x - options.patch_size/2 : x + options.patch_size/2,
                                 y - options.patch_size/2 : y + options.patch_size/2]
#                cangle = 2.0*np.pi*np.random.rand()
                cangle = np.pi/4
                cpatchsz = options.patch_size
                matrix = np.array([[ np.cos(cangle), np.sin(cangle), 0],
                                   [-np.sin(cangle), np.cos(cangle), 0],
                                   [              0,              0, 1]],dtype=np.float64)
                offset = np.zeros((3,),dtype=np.float64)
                offset[0] = float(cpatchsz)/2.0 - 0.5
                offset[1] = float(cpatchsz)/2.0 - 0.5
                offset = np.dot(matrix,offset)
                offset[0] = x - 0.5 - offset[0]
                offset[1] = y - 0.5 - offset[1]
                cpatch = spnd.interpolation.affine_transform(satim,matrix,offset,(cpatchsz,cpatchsz,3))

#                mapVal = mapim[x + options.patch_size/2,y + options.patch_size/2]
#                imData[:,i] = satPatch.flat
#                mapData[i] = mapVal != 0
#                i += 1
                plt.figure(1)
                plt.imshow(satPatch)
                plt.figure(2)
                plt.imshow(cpatch)
                plt.show()
#        print 'i = {0}, num_patches = {1}\n'.format(i,num_patches)
#        np.save('{0}_{1}_data.npy'.format(outfile_base,file_ind),imData[:,:i])
#        np.save('{0}_{1}_labels.npy'.format(outfile_base,file_ind),imData[:,:i])


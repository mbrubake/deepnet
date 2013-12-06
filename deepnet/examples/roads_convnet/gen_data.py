#!/usr/bin/python
import time, os, random, bz2, gzip
import numpy as np
from scipy import ndimage as spnd
from scipy.ndimage import morphology as morph
from scipy import misc
from glob import glob
from optparse import OptionParser
import matplotlib.pyplot as plt
from itertools import product

class RunningStats:
    def __init__(self,ndim,dtype=np.float64):
        self.n = 0
        self.mean = np.zeros((ndim),dtype=dtype)
        self.M2 = np.zeros((ndim),dtype=dtype)

    def update(self,data):
        weight = data.shape[0]
        x = np.mean(data,axis=0)
        newn = weight + self.n
        delta = x - self.mean
        R = delta * (float(weight) / float(newn))
        self.mean += R
        self.M2 += self.n * np.multiply(delta,R)
        self.n = newn
 
    def get_mean(self):
        return self.mean

    def get_std(self):
        return np.sqrt(self.M2/(self.n - 1))

    def write_stats(self,filename):
        stats = dict()
        stats['n'] = self.n
        stats['mean'] = self.get_mean()
        stats['std'] = self.get_std()
        np.savez(filename,**stats)
 
if __name__ == '__main__':
#if 0:
    cmdLineParser = OptionParser()
    cmdLineParser.add_option('--sat_patch_size',type='int',dest='sat_patch_size',default=64)
    cmdLineParser.add_option('--map_patch_size',type='int',dest='map_patch_size',default=16)
    cmdLineParser.add_option('--dilate_map',type='int',dest='dilate_map',default=0)
    cmdLineParser.add_option('--patch_stride',type='int',dest='patch_stride',default=16)
    cmdLineParser.add_option('--num_batches',type='int',dest='num_batches',default=1)
    cmdLineParser.add_option('--set_start_num',type='int',dest='set_start_num',default=1)
    cmdLineParser.add_option('--patches_per_file',type='int',dest='patches_per_file',default=2**14)
    cmdLineParser.add_option('--randrot',type='int',dest='randrot',default=False)
    cmdLineParser.add_option('--randoff',type='int',dest='randoff',default=False)
    cmdLineParser.add_option('--randorder',type='int',dest='randorder',default=False)
    cmdLineParser.add_option('--patch_norm',type='int',dest='patch_norm',default=True)
    cmdLineParser.add_option('--output',type='string',dest='output',default='npgz')
    cmdLineParser.add_option('--stats_file',type='string',dest='stats_file',default=None)

    (options, args) = cmdLineParser.parse_args()
    indir = args[0];
    outdir = args[1];

    print 'Loading images from {0}\nWritting patch data to {1}'.format(indir,outdir)

    sat_filelist = sorted(glob(os.path.join(indir,'sat','*.tiff')))
    map_filelist = sorted(glob(os.path.join(indir,'map','*.tif')))
    fileList = [(satimFile,mapimFile) for (satimFile,mapimFile) in zip(sat_filelist,map_filelist)]

    maxpatchsize = max((options.sat_patch_size,options.map_patch_size))
    if options.randrot:
        maxunrotpatchsz = int(np.ceil(np.sqrt(2.0)*maxpatchsize))
        maxunrotpatchsz = maxunrotpatchsz + (maxunrotpatchsz%2)
    else:
        maxunrotpatchsz = maxpatchsize

    nSatPatchPts = options.sat_patch_size**2
    nMapPatchPts = options.map_patch_size**2
    sat_patchPts = np.mgrid[-options.sat_patch_size/2:options.sat_patch_size/2,
                            -options.sat_patch_size/2:options.sat_patch_size/2].reshape((2,nSatPatchPts))
    map_patchPts = np.mgrid[-options.map_patch_size/2:options.map_patch_size/2,
                            -options.map_patch_size/2:options.map_patch_size/2].reshape((2,nMapPatchPts))
 
    map_patchset = np.empty((options.patches_per_file,nMapPatchPts),dtype=np.uint8)
    sat_patchset = np.empty((options.patches_per_file,3*nSatPatchPts),dtype=np.float32)
    nextPatchSetNum = 0
    currSetNum = options.set_start_num

    num_outputs = 0
    num_road_outputs = 0

    totalNumDataPts = 0

    if options.stats_file:
        sat_stats = RunningStats(3*nSatPatchPts)


    for batchnum in range(options.num_batches):
        if options.randorder:
            # Shuffle files
            random.shuffle(fileList)

        for (file_ind,(satimFile,mapimFile)) in enumerate(fileList):
            assert os.path.splitext(os.path.split(satimFile)[1])[0] == os.path.splitext(os.path.split(mapimFile)[1])[0]
#            print '\nImage: {0}'.format(os.path.split(satimFile)[1])
            satim = (misc.imread(satimFile).astype('float32') / 255.0) - 0.5
            valim = np.logical_not(\
                        morph.binary_dilation(\
                            morph.binary_erosion(np.all(satim > 253.0/255.0,axis=2),iterations=8),iterations=8+maxpatchsize/2))
            mapim = misc.imread(mapimFile)
            if options.dilate_map:
                mapim = morph.binary_dilation(mapim,iterations=options.dilate_map)
            mapim = mapim.astype('float32') / 255.0

            # Update statistics
            num_outputs += np.sum(valim)
            num_road_outputs += np.sum(mapim[valim])

            if options.randoff:
                startx = np.random.randint(maxunrotpatchsz/2,maxpatchsize+1)
                starty = np.random.randint(maxunrotpatchsz/2,maxpatchsize+1)
            else:
                startx = maxunrotpatchsz/2
                starty = maxunrotpatchsz/2

            range1 = range(startx,satim.shape[0] - maxunrotpatchsz/2 + 1,options.patch_stride)
            range2 = range(starty,satim.shape[1] - maxunrotpatchsz/2 + 1,options.patch_stride)
            patchCenters = np.array([(x,y) for (x,y) in product(range1,range2) if valim[x,y]]).T
            nPatches = patchCenters.shape[1]

            sat_patches = np.empty((nSatPatchPts,satim.shape[2],nPatches),dtype=np.float32)
            map_patches = np.empty((nMapPatchPts,nPatches),dtype=np.float32)

            if options.randrot:
                angles = (2.0*np.pi)*np.random.rand(nPatches)
                rots = np.array([[ np.cos(angles), np.sin(angles)],
                                 [-np.sin(angles), np.cos(angles)]],dtype=np.float32)

                map_patchCoords = np.dot(rots.transpose((2,0,1)), \
                                     map_patchPts.reshape((1,2,nMapPatchPts))).reshape((nPatches,2,nMapPatchPts)) \
                                                                           .transpose((1,2,0)) \
                                  + patchCenters[:,0:nPatches].reshape((2,1,nPatches)) + 0.5
                sat_patchCoords = np.dot(rots.transpose((2,0,1)), \
                                     sat_patchPts.reshape((1,2,nSatPatchPts))).reshape((nPatches,2,nSatPatchPts)) \
                                                                           .transpose((1,2,0)) \
                                  + patchCenters[:,0:nPatches].reshape((2,1,nPatches)) + 0.5
                for cdim in range(0,satim.shape[2]):
                    sat_patches[:,cdim,:] = spnd.interpolation.map_coordinates(satim[:,:,cdim].T,sat_patchCoords,order=2)
                spnd.interpolation.map_coordinates(mapim.T,map_patchCoords,output=map_patches,order=0)
            else:
                map_patchCoords = map_patchPts.reshape((2,nMapPatchPts,1)) + patchCenters[:,0:nPatches].reshape((2,1,nPatches))
                sat_patchCoords = sat_patchPts.reshape((2,nSatPatchPts,1)) + patchCenters[:,0:nPatches].reshape((2,1,nPatches))
                for cdim in range(0,satim.shape[2]):
                    sat_patches[:,cdim,:] = satim[sat_patchCoords[0,...],sat_patchCoords[1,...],cdim]
                map_patches[:,:] = mapim[map_patchCoords[0,...],map_patchCoords[1,...]]

            if options.patch_norm:
                # Mean normalize satellite patches
                sat_patches -= np.mean(sat_patches,axis=(0,1)).reshape((1,1,nPatches))

            # Add patches to patch set
            nCurrPatches = min(nPatches,options.patches_per_file - nextPatchSetNum)
            map_patchset[nextPatchSetNum:nextPatchSetNum+nCurrPatches,:] = map_patches[:,:nCurrPatches].reshape((-1,nCurrPatches)).T
            sat_patchset[nextPatchSetNum:nextPatchSetNum+nCurrPatches,:] = sat_patches[:,:,:nCurrPatches].reshape((-1,nCurrPatches)).T
            nextPatchSetNum += nCurrPatches

            if options.stats_file:
                sat_stats.update(sat_patches.reshape((-1,nPatches)).T)
                sat_stats.write_stats(options.stats_file)

            if nextPatchSetNum == options.patches_per_file or (batchnum == options.num_batches-1 and file_ind == len(fileList)-1):
                print 'Saving set number {0} (so far: {1} patches, {2}% roads)'.format(currSetNum, totalNumDataPts + nextPatchSetNum, 100.0*num_road_outputs/num_outputs)
                if options.randorder:
                    out_map = map_patchset[np.random.permutation(nextPatchSetNum),:]
                    out_sat = sat_patchset[np.random.permutation(nextPatchSetNum),:]
                else:
                    out_map = map_patchset[:nextPatchSetNum,:]
                    out_sat = sat_patchset[:nextPatchSetNum,:]

                if options.output == 'npgz':
                    out_map_filename = '{0}/map_patches_{1:0=10}.npgz'.format(outdir,currSetNum)
                    out_sat_filename = '{0}/sat_patches_{1:0=10}.npgz'.format(outdir,currSetNum) 
                    np.save(gzip.GzipFile(out_map_filename,'wb'),out_map)
                    np.save(gzip.GzipFile(out_sat_filename,'wb'),out_sat)
                elif options.output == 'npbz2':
                    out_map_filename = '{0}/map_patches_{1:0=10}.npbz2'.format(outdir,currSetNum)
                    out_sat_filename = '{0}/sat_patches_{1:0=10}.npbz2'.format(outdir,currSetNum) 
                    np.save(bz2.BZ2File(out_map_filename,'wb'),out_map)
                    np.save(bz2.BZ2File(out_sat_filename,'wb'),out_sat)
                elif options.output == 'npy':
                    out_map_filename = '{0}/map_patches_{1:0=10}.npy'.format(outdir,currSetNum)
                    out_sat_filename = '{0}/sat_patches_{1:0=10}.npy'.format(outdir,currSetNum) 
                    np.save(out_map_filename,out_map)
                    np.save(out_sat_filename,out_sat)
                else:
                    assert False, 'Unknown output type: ' + options.output

                currSetNum += 1

                totalNumDataPts += nextPatchSetNum

                nextPatchSetNum = 0
                if nPatches > nCurrPatches:
                    map_patchset[:(nPatches-nCurrPatches),:] = map_patches[:,nCurrPatches:].reshape((-1,nPatches-nCurrPatches)).T
                    sat_patchset[:(nPatches-nCurrPatches),:] = sat_patches[:,:,nCurrPatches:].reshape((-1,nPatches-nCurrPatches)).T
                    nextPatchSetNum = nPatches-nCurrPatches

    print 'Dumped {0} sets with {1} datapoints to {2}'.format(currSetNum-1,totalNumDataPts,outdir)

    raw_input('Waiting to exit...')


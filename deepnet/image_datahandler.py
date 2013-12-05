from scipy import misc
from scipy import ndimage as spnd
from itertools import product
import numpy as np

class DiskImage(object):
    def __init__(self, imgnames, validimgnames, numdim_list, patchsize, patchstride, normalization, rotation, randoff):
        assert len(imgnames) == len(validimgnames)
        assert len(imgnames) == len(numdim_list)
        assert len(imgnames) == len(patchsize)
        assert len(imgnames) == len(normalization)
        self.num_data = len(imgnames)
        self.imgnames = imgnames
        self.validimgnames = validimgnames
        self.patchsize = patchsize
        self.patchstride = patchstride
        self._num_file_list = [len(filename_list) for filename_list in imgnames]
        self._maxpos = 1e300  # Total amount of data, set to arbitrarily large number
        self.maxpatchsize = np.max(self.patchsize)
        self.normalization = normalization
        self.rotation = rotation
        self.randoff = randoff

        self.numdim_list = [None]*self.num_data
        self.data = [None]*self.num_data
        self.last_read_patch = [None]*self.num_data
        self.last_read_image = [None]*self.num_data
        self.last_read_valid= [None]*self.num_data
        self.last_read_imgnum = [-1]*self.num_data
        self.last_read_img_npatches = [-1]*self.num_data
        current_file = 0
        if self.rotation:
            maxunrotpatchsz = int(np.ceil(np.sqrt(2.0)*self.maxpatchsize))
            maxunrotpatchsz = maxunrotpatchsz + (maxunrotpatchsz%2)
        else:
            maxunrotpatchsz = self.maxpatchsize
        for i in range(self.num_data):
            curr_psize = patchsize[i]
            assert (curr_psize%2) == 0, 'Patch sizes must be multiples of 2'
            self.LoadImage(i)

            print '  Total number of examples: {1}*{2} = {0}'.format(len(self.imgnames[i])*self.last_read_img_npatches[i],
                                                                         len(self.imgnames[i]),self.last_read_img_npatches[i])
            if self.last_read_image[i].ndim == 2:
                self.numdim_list[i] = curr_psize**2
            else:
                self.numdim_list[i] = self.last_read_image[i].shape[2] * curr_psize**2
            
            assert(self.numdim_list[i] == numdim_list[i])
            
    def LoadImage(self,i,current_file = None):
        if current_file == None:
            current_file = self.last_read_imgnum[i]

        if self.validimgnames[i] != None and self.validimgnames[i][current_file] != None:
            valid_image = misc.imread(self.validimgnames[i][current_file])
        else:
            valid_image = None

        this_image = misc.imread(self.imgnames[i][current_file])

        if this_image.ndim == 2:
            this_image = this_image.reshape(this_image.shape + (1,))

        if this_image.dtype == 'uint8':
            this_image = this_image.astype('float32') / 255.0

        if self.rotation:
            maxunrotpatchsz = int(np.ceil(np.sqrt(2.0)*self.maxpatchsize))
            maxunrotpatchsz = maxunrotpatchsz + (maxunrotpatchsz%2)
        else:
            maxunrotpatchsz = self.maxpatchsize
 
        self.last_read_image[i] = this_image
        self.last_read_valid[i] = valid_image
        self.last_read_imgnum[i] = current_file
        if self.randoff:
            startx = np.random.randint(maxunrotpatchsz/2,self.maxpatchsize+1)
            starty = np.random.randint(maxunrotpatchsz/2,self.maxpatchsize+1)
        else:
            startx = maxunrotpatchsz/2
            starty = maxunrotpatchsz/2
        range1 = range(startx,this_image.shape[0] - maxunrotpatchsz/2 + 1,self.patchstride)
        range2 = range(starty,this_image.shape[1] - maxunrotpatchsz/2 + 1,self.patchstride)
        self.last_read_patch[i] = product(range1,range2)
        self.last_read_img_npatches[i] = len(range1)*len(range2)
 
    def Get(self, batchsize):
        data_list = []
        if self.rotation:
            angles = (2.0*np.pi)*np.random.rand(batchsize)
            rots = np.array([[ np.cos(angles), np.sin(angles)],
                             [-np.sin(angles), np.cos(angles)]],dtype=np.float32)
        patchCenters = np.empty((2,batchsize),dtype=np.int32)
        for i in range(self.num_data):
            imgnames = self.imgnames[i]
            numdims = self.numdim_list[i]
            num_files = self._num_file_list[i]
            current_file = self.last_read_imgnum[i]
            cnorm = self.normalization[i]
            cpatchsz = self.patchsize[i]

            nPatchPts = cpatchsz**2
            patchPts = np.mgrid[-cpatchsz/2:cpatchsz/2,
                                -cpatchsz/2:cpatchsz/2].reshape((2,nPatchPts))
          
            if self.data[i] is None or self.data[i].shape[0] != batchsize:
                self.data[i] = np.zeros((batchsize, numdims), dtype='float32')
            data = self.data[i]
            
            # Read data from disk.
            datasize = 0  # Number of rows of data filled up.
            while(datasize < batchsize):
                # Load the next image if needed
                self.LoadImage(i,current_file)

                this_image = self.last_read_image[i]
                valid_image = self.last_read_valid[i]

                # Get the patch center coordinates
                nPatches = 0
                for (x,y) in self.last_read_patch[i]:
                    if valid_image == None or valid_image[x,y]:
                        patchCenters[0,nPatches] = x
                        patchCenters[1,nPatches] = y
                        nPatches += 1
                        if nPatches == patchCenters.shape[1] or nPatches == batchsize - datasize:
                            break

                if nPatches == 0:
                    current_file = (current_file + 1) % num_files
                    continue
                

                patches = np.empty((this_image.shape[2],nPatchPts,nPatches),dtype=np.float32)
                if self.rotation:
                    patchCoords = np.dot(rots[:,:,datasize:datasize+nPatches].transpose((2,0,1)), \
                                         patchPts.reshape((1,2,nPatchPts))).reshape((nPatches,2,nPatchPts)) \
                                                                           .transpose((1,2,0)) \
                                  + patchCenters[:,0:nPatches].reshape((2,1,nPatches)) + 0.5

                    for cdim in range(0,this_image.shape[2]):
                        spnd.interpolation.map_coordinates(this_image[:,:,cdim].T,patchCoords,output=patches[cdim,:,:],order=0)
                else:
                    patchCoords = patchPts.reshape((2,nPatchPts,1)) + patchCenters[:,0:nPatches].reshape((2,1,nPatches))
                    for cdim in range(0,this_image.shape[2]):
                        patches[cdim,:,:] = this_image[patchCoords[0,...],patchCoords[1,...],cdim]


                if cnorm:
                    patches -= np.mean(np.mean(patches,axis=1),axis=0).reshape((1,1,nPatches))

                data[datasize:datasize+nPatches,:] = patches.transpose((2,1,0)).reshape((nPatches,-1))
                datasize += nPatches
                if datasize >= batchsize:
                    break
                current_file = (current_file + 1) % num_files

            data_list.append(data)
        return data_list





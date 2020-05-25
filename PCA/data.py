#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# DOWNLOAD:
# curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
# curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# USAGE:
# TrD = TrainData() # or TestData
# X, y = TrD[0:1000] # get first 1000 images
# X, y = TrD[:] # get 'em all

import gzip
import numpy as np

class Data:
    _images_file = None
    _labels_file = None
    def __init__(self):
        self._cur = 0
        if not self._images_file or not self._labels_file:
            raise Exception(f"Invalid files: {self._images_file} & {self._labels_file}")
        self._images_fh = gzip.open(self._images_file, 'r')
        self._labels_fh = gzip.open(self._labels_file, 'r')
        _, iimages, self.rows, self.cols = [
            int.from_bytes(self._images_fh.read(4), byteorder='big') for _ in range(4)
        ]
        _, limages = [
            int.from_bytes(self._labels_fh.read(4), byteorder='big') for _ in range(2)
        ]
        assert iimages == limages
        self.images = iimages
        
    def __len__(self):
        return self.images
    
    def __getitem__(self, slices):
        rest = tuple()
        if type(slices) == tuple:
            slices, *rest = slices
        if type(slices) == int:
            slices = [slices]
        elif type(slices) == slice:
            slices = range(*slices.indices(len(self)))
            
        ibuf, lbuf = b'', b''
        ifh, lfh = self._images_fh, self._labels_fh
        for image in slices:
            if image - self._cur != 0:
                ifh.seek(self.rows * self.cols * (image - self._cur), 1)
                lfh.seek(image - self._cur, 1)
            ibuf += ifh.read(self.rows * self.cols)
            lbuf += lfh.read(1)
            self._cur = image + 1
        
        X = np.frombuffer(ibuf, dtype=np.uint8)               .reshape(len(slices), self.rows, self.cols)
        
        y = np.frombuffer(lbuf, dtype=np.uint8)
        
        return X[rest], y
    
class TrainData(Data):
    _images_file = 'train-images-idx3-ubyte.gz'
    _labels_file = 'train-labels-idx1-ubyte.gz'
    
class TestData(Data):
    _images_file = 't10k-images-idx3-ubyte.gz'
    _labels_file = 't10k-labels-idx1-ubyte.gz'
    


#!/usr/bin/env python
#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

import random, copy, os
import scipy, numpy as np
from keras.preprocessing.image import ImageDataGenerator
import gc

import hrl_lib.util as ut
import cv2



random.seed(3334)
np.random.seed(3334)


class myGenerator():
    ''' image + signal data augmentation '''
    
    def __init__(self, augmentation=False, rescale=1.0):

        if augmentation:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                rescale=rescale,
                width_shift_range=0.05,
                height_shift_range=0.05,
                zoom_range=0.1,
                horizontal_flip=False,
                fill_mode='nearest',
                dim_ordering="th")
        else:
            self.datagen = ImageDataGenerator(
                rescale=rescale,
                dim_ordering="th")
        pass

    def flow(self, x_img, x_sig, y, batch_size=32, shuffle=True):

        assert len(x_img) == len(y), "data should have the same length"
        assert len(x_img) == len(x_sig), "data should have the same length"
        
        if type(x_img) is not np.ndarray: x_img = np.array(x_img)
        if type(x_sig) is not np.ndarray: x_sig = np.array(x_sig)
        if type(y) is not np.ndarray: y = np.array(y)
        
        while 1:
            idx_list = range(len(x_img))
            if shuffle: random.shuffle(idx_list)
            
            x_sig_new = x_sig[idx_list]
            i = -1

            for x_batch_img, y_batch in self.datagen.flow(x_img[idx_list], y[idx_list],
                                                          batch_size=batch_size, shuffle=False):                
                if (i+1)*batch_size >= len(y):
                    break
                else:
                    i += 1
                    yield [x_batch_img, x_sig_new[i*batch_size:(i+1)*batch_size]], y_batch

            gc.collect()
            
            ## for i in range(1875): # 1875 * 32 = 60000 -> # of training samples
            ##     if i%125==0:
            ##         print "i = " + str(i)
            ##     yield X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]


class sigGenerator():
    ''' signal data augmentation
    Signals should be normalized before.
    '''
    
    def __init__(self, augmentation=False, noise_mag=0.05):

        self.augmentation  = augmentation
        self.noise_mag     = noise_mag
        pass

    def flow(self, x, y, batch_size=32, shuffle=True):

        assert len(x) == len(y), "data should have the same length"
        
        if type(x) is not np.ndarray: x = np.array(x)
        if type(y) is not np.ndarray: y = np.array(y)
        
        while 1:

            idx_list = range(len(x))
            if shuffle: random.shuffle(idx_list)
            
            x_new = copy.copy(x[idx_list])
            y_new = y[idx_list]

            if self.augmentation:
                x_new += np.random.normal(0.0, self.noise_mag, \
                                          np.shape(x_new)) 
            
            for idx in range(0,len(y_new), batch_size):
                if idx >= len(y_new):
                    break
                else:
                    yield x_new[idx:idx+batch_size], y_new[idx:idx+batch_size]

            gc.collect()



def extract_hypercolumn(model, layer_indexes, instance):
    import theano
    import scipy as sp
    
    layers = [model.layers[li].output for li in layer_indexes]
    ## layers = [model.layers[li].get_output(train=False) for li in layer_indexes]
    get_feature = theano.function([model.layers[0].input], layers,
                                  allow_input_downcast=False)
    feature_maps = get_feature(instance)
    hypercolumns = []
    for convmap in feature_maps:
        for fmap in convmap[0]:
            upscaled = sp.misc.imresize(fmap, size=(224, 224),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
            
    return np.asarray(hypercolumns)




if __name__ == '__main__':

    n = 102
    imgs = None
    for i in xrange(n):
        img = np.array([[np.ones((10,10)).tolist(),
                         np.ones((10,10)).tolist(),
                         np.ones((10,10)).tolist()]])+i
        
        if imgs is None: imgs = img
        else: imgs = np.vstack([imgs, img])
            
    ## datagen = myGenerator(True)            
    ## count = 0
    ## for x,y in datagen.flow(imgs.astype(float), imgs.astype(float), range(n), batch_size=5, shuffle=False):
    ##     (x1,x2) = x
    ##     print x1[:,0,0,0], x2[:,0,0,0] #, x2.shape
    ##     count +=1
    ##     if count > 99: break

    print np.shape(imgs), i

    datagen = sigGenerator(True)            
    count = 0
    for x,y in datagen.flow(imgs.astype(float), range(n), batch_size=5, shuffle=False):
        print x[:,0,0,0] #, x2.shape
        count +=1
        if count > 99: break

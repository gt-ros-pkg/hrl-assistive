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
from scipy.stats import truncnorm
import gc

import hrl_lib.util as ut
import cv2



random.seed(3334)
np.random.seed(3334)


class sigGenerator():
    ''' signal data augmentation
    Signals should be normalized before.
    '''
    
    def __init__(self, augmentation=False, noise_mag=0.03):

        self.augmentation  = augmentation
        self.noise_mag     = noise_mag
        self.total_batches_seen = 0
        self.batch_index = 0

    def reset(self):
        self.batch_index = 0

    def flow(self, x, y, batch_size=32, shuffle=True, seed=None):

        assert len(x) == len(y), "data should have the same length"
        
        if type(x) is not np.ndarray: x = np.array(x)
        if type(y) is not np.ndarray: y = np.array(y)

        # Ensure self.batch_index is 0.
        self.reset()
        x_new = x[:]

        # TODO: Need to add random selection
        if len(x_new) % batch_size > 0:
            n_add = batch_size - len(x_new) % batch_size
            x_new = np.vstack([x_new, x_new[:n_add] ])

        n = len(x_new)
        idx_list = range(n)
        n_dim = len(x_new[0,0])

        while 1:

            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)

            if self.batch_index == 0:
                idx_list = range(n)
                if shuffle:
                    random.shuffle(idx_list)
                x_new = x_new[idx_list]

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0

            ## print " aa ", batch_size, " aa ", self.total_batches_seen, self.batch_index, current_index,"/",n
            
            self.total_batches_seen += 1
            if self.augmentation:
                # noise
                noise = np.random.normal(0.0, self.noise_mag, \
                                          np.shape(x_new[current_index:current_index+current_batch_size]))

                '''                                          
                # shift? (left-right)
                for i in range(current_batch_size):
                    xx = x_new[current_index+i]
                    idx_offset = int(np.round(np.random.normal(0,1, size=1))[0])

                    if idx_offset>=0:
                        xx = xx[idx_offset:]
                        x_new[current_index+i] = np.pad(xx, ((0,idx_offset),
                                                            (0,0)),
                                                            mode='edge')
                    else:
                        xx = xx[:idx_offset]
                        x_new[current_index+i] = np.pad(xx, ((abs(idx_offset),0),
                                                            (0,0)),
                                                            mode='edge')
                '''

                # scaling                

                
                # up-down
                ## for i in range(current_batch_size):
                ##     ud_offset = np.random.normal(0.0, 0.02, size=(n_dim)) 
                ##     for j in range(n_dim):
                ##         x_new[current_index+i,:,j] += ud_offset[j]
                     
                yield x_new[current_index:current_index+current_batch_size]+noise,\
                  x_new[current_index:current_index+current_batch_size]+noise
                                          
            else:
                yield x_new[current_index:current_index+current_batch_size],\
                  x_new[current_index:current_index+current_batch_size]


                
            ## for idx in range(0,len(x_new), batch_size):
            ##     if idx >= len(x_new):
            ##         break
            ##     else:
            ##         self.total_batches_seen += 1
            ##         yield x_new[idx:idx+batch_size], x_new[idx:idx+batch_size]

            ## gc.collect()

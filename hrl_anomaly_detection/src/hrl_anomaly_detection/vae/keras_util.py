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
        pass

    def flow(self, x, y, batch_size=32, shuffle=True):

        assert len(x) == len(y), "data should have the same length"
        
        if type(x) is not np.ndarray: x = np.array(x)
        if type(y) is not np.ndarray: y = np.array(y)
        
        while 1:

            idx_list = range(len(x))
            if shuffle: random.shuffle(idx_list)
            
            x_new = copy.copy(x[idx_list])

            if self.augmentation:
                x_new += np.random.normal(0.0, self.noise_mag, \
                                          np.shape(x_new)) 
            
            for idx in range(0,len(x_new), batch_size):
                if idx >= len(x_new):
                    break
                else:
                    yield x_new[idx:idx+batch_size], x_new[idx:idx+batch_size]

            gc.collect()

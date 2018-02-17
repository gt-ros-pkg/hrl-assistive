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

from keras.preprocessing.image import ImageDataGenerator
import hrl_lib.util as ut
import cv2



random.seed(3334)
np.random.seed(3334)


class multiGenerator():
    ''' image + signal data augmentation '''
    
    def __init__(self, augmentation=False, noise_mag=0.03, rescale=1.0):

        self.noise_mag     = noise_mag
        self.augmentation  = augmentation
        self.total_batches_seen = 0
        self.batch_index = 0

        if augmentation:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                rescale=rescale,
                width_shift_range=0.05,
                height_shift_range=0.05,
                zoom_range=0.1,
                horizontal_flip=False,
                fill_mode='nearest',
                data_format="channels_first")
        else:
            self.datagen = ImageDataGenerator(
                rescale=rescale,
                data_format="channels_first")
        pass

    def reset(self):
        self.batch_index = 0
        
    def flow(self, x_img, x_sig, y, batch_size=32, shuffle=True, seed=None):

        assert len(x_img) == len(y), "data should have the same length"
        assert len(x_img) == len(x_sig), "data should have the same length"
        
        if type(x_img) is not np.ndarray: x_img = np.array(x_img)
        if type(x_sig) is not np.ndarray: x_sig = np.array(x_sig)
        if type(y) is not np.ndarray: y = np.array(y)

        # Ensure self.batch_index is 0.
        self.reset()
        x_sig_new = x_sig[:]
        x_img_new = x_img[:]
        y_new = y[:]

        # TODO: Need to add random selection
        if len(y_new) % batch_size > 0:
            n_add = batch_size - len(y_new) % batch_size
            x_sig_new = np.vstack([x_sig_new, x_sig_new[:n_add] ])
            x_img_new = np.vstack([x_img_new, x_img_new[:n_add] ])
            y_new = np.vstack([y_new, y_new[:n_add] ])
        

        n = len(x_img)
            
        while 1:

            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            
            if self.batch_index == 0:
                idx_list = range(n)
                if shuffle: random.shuffle(idx_list)
                x_sig_new = x_sig_new[idx_list]
                x_img_new = x_img_new[idx_list]
                y_new = y_new[idx_list]

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0

            self.total_batches_seen += 1

            i = -1
            for x_batch_img, y_batch in self.datagen.flow(x_img_new[idx_list], y_new[idx_list],
                                                          batch_size=batch_size, shuffle=False):

                if x_batch_img is None:
                    sys.exit()

                if (i+1)*batch_size >= len(y_new):
                    break
                else:
                    i += 1
                    x = x_sig_new[i*batch_size:(i+1)*batch_size]
                    if self.augmentation:
                        noise = np.random.normal(0.0, self.noise_mag, np.shape(x))
                        yield [x_batch_img, x+noise], y_batch
                    else:
                        yield [x_batch_img, x], y_batch
                    
                gc.collect()


class sigGenerator():
    ''' signal data augmentation
    Signals should be normalized before.
    '''
    
    def __init__(self, augmentation=False, noise_mag=0.03, add_phase=False):

        self.augmentation  = augmentation
        self.noise_mag     = noise_mag
        self.add_phase     = add_phase
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
        y_new = y[:]

        # TODO: Need to add random selection
        if len(x_new) % batch_size > 0:
            n_add = batch_size - len(x_new) % batch_size
            x_new = np.vstack([x_new, x_new[:n_add] ])
            y_new = np.vstack([y_new, y_new[:n_add] ])

        n = len(x_new)
        idx_list = range(n)
        n_dim  = np.shape(x_new)[-1]
        length = len(x_new[0])

        while 1:

            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)

            if self.batch_index == 0:
                idx_list = range(n)
                if shuffle:
                    random.shuffle(idx_list)
                x_new = x_new[idx_list]
                y_new = y_new[idx_list]

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0

            self.total_batches_seen += 1
            if self.augmentation:
                # noise
                noise = np.random.normal(0.0, self.noise_mag, \
                                          np.shape(x_new[current_index:current_index+current_batch_size]))

                yield x_new[current_index:current_index+current_batch_size]+noise,\
                  y_new[current_index:current_index+current_batch_size]
                                          
            else:
                yield x_new[current_index:current_index+current_batch_size],\
                  y_new[current_index:current_index+current_batch_size]



def get_bottleneck_image(save_data_path, n_labels, fold_list, vgg=True, use_extra_img=True,
                         remove_label=[]):

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    for idx in fold_list:

        # Loading data
        from hrl_execution_monitor import util as autil
        train_data, test_data = autil.load_data(idx, save_data_path, extra_img=use_extra_img, viz=False)      
        x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_img = test_data[1]
        y_test     = test_data[2]

        # remove specific label --------------------
        add_idx = []
        for i, y in enumerate(y_train):
            if np.argmax(y) not in remove_label:
                add_idx.append(i)
        x_train_img = np.array(x_train_img)[add_idx]
        y_train = np.array(y_train)[add_idx]

        add_idx = []
        for i, y in enumerate(y_test):
            if np.argmax(y) not in remove_label:
                add_idx.append(i)
        x_test_img = np.array(x_test_img)[add_idx]
        y_test = np.array(y_test)[add_idx]
        #--------------------------------------------
        
        if vgg: model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels)
        else: model = km.cnn_net(np.shape(x_train_img)[1:], n_labels)            

        bt_data_path = os.path.join(save_data_path, 'bt')
        if os.path.isdir(bt_data_path) is False:
            os.system('mkdir -p '+bt_data_path)
            
        # ------------------------------------------------------------
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest',
            dim_ordering="th")

        x_ = None
        y_ = None
        count = 0
        for x_batch, y_batch in train_datagen.flow(x_train_img, y_train, batch_size=len(x_train_img),
                                                   shuffle=False):
            if x_ is None: x_ = model.predict(x_batch)
            else: x_ = np.vstack([x_, model.predict(x_batch)])
            if y_ is None: y_ = y_train
            else: y_ = np.vstack([y_, y_train])
            count += 1
            print count
            if count > 4: break

        np.save(open(os.path.join(bt_data_path,'x_train_bt_'+str(idx)+'.npy'), 'w'), x_)
        np.save(open(os.path.join(bt_data_path,'y_train_bt_'+str(idx)+'.npy'), 'w'), y_)
        del x_, y_, train_datagen

        # ------------------------------------------------------------
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=False,
            dim_ordering="th")

        x_ = None
        y_ = None
        count = 0
        for x_batch, y_batch in test_datagen.flow(x_test_img, y_test, batch_size=len(x_test_img), shuffle=False):
            if x_ is None: x_ = model.predict(x_batch)
            else: x_ = np.vstack([x_, model.predict(x_batch)])
            if y_ is None: y_ = y_test
            else: y_ = np.vstack([y_, y_test])
            count += 1
            print count
            if count > 0: break

        np.save(open(os.path.join(bt_data_path,'x_test_bt_'+str(idx)+'.npy'), 'w'), x_)
        np.save(open(os.path.join(bt_data_path,'y_test_bt_'+str(idx)+'.npy'), 'w'), y_)
        del x_, y_, test_datagen
        
        gc.collect()

    return

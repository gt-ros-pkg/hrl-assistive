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
                width_shift_range=0.2,
                height_shift_range=0.2,
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



def preprocess_data(src_pkl, save_data_path, img_scale=0.25, nb_classes=12,
                    img_feature_type='cnn', verbose=False):
    ''' Preprocessing data '''

    save_data_path = os.path.join(save_data_path, 'keras')
    if os.path.isdir(save_data_path) is False:
        os.system('mkdir -p '+save_data_path)
       
    d = ut.load_pickle(src_pkl)
    nFold = len(d.keys())

    for idx in xrange(nFold):

        (x_trains, y_train, x_tests, y_test) = d[idx]         
        x_train_sig = x_trains[0] #signals
        x_test_sig  = x_tests[0]
        x_train_img = x_trains[1] #img
        x_test_img  = x_tests[1]
        if verbose:
            print "Training data: ", np.shape(x_train_img), np.shape(x_train_sig),
            np.shape(y_train)
            print "Testing data: ", np.shape(x_test_img), np.shape(x_test_sig),
            np.shape(y_test)

        start_label = np.unique(np.array(y_train).flatten().tolist()+np.array(y_test).flatten().tolist())
        start_label = np.amin(start_label)
        print "start label", start_label

        #--------------------------------------------------------------------
        # check images
        for ii in xrange(2):
            if ii == 0:
                x_img = x_train_img
                x_sig = x_train_sig
                y     = y_train
            else:
                x_img = x_test_img
                x_sig = x_test_sig
                y     = y_test
            
            rm_idx = []
            x = []
            for j, f in enumerate(x_img):
                if f is None:
                    print "None image ", j+1, '/', len(x_img)
                    rm_idx.append(j)
                    continue

                img = extract_image(f, img_feature_type=img_feature_type,
                                    img_scale=img_scale)
                x.append(img)
            x_img = x

            # check signals and labels
            x_sig = [x_sig[i] for i in xrange(len(x_sig)) if i not in rm_idx ]
            y = [y[i] for i in xrange(len(y)) if i not in rm_idx ]
            y = np.array(y)-start_label # make label start from zero
            if img_feature_type == 'cnn' or img_feature_type == 'vgg':
                from keras.utils.np_utils import to_categorical
                y = to_categorical(y, nb_classes=nb_classes)

            if ii == 0:
                np.save(open(os.path.join(save_data_path,'x_train_img_'+str(idx)+'.npy'), 'w'), x_img)
                np.save(open(os.path.join(save_data_path,'x_train_sig_'+str(idx)+'.npy'), 'w'), x_sig)
                np.save(open(os.path.join(save_data_path,'y_train_'+str(idx)+'.npy'), 'w'), y)
            else:
                np.save(open(os.path.join(save_data_path,'x_test_img_'+str(idx)+'.npy'), 'w'), x_img)
                np.save(open(os.path.join(save_data_path,'x_test_sig_'+str(idx)+'.npy'), 'w'), x_sig)
                np.save(open(os.path.join(save_data_path,'y_test_'+str(idx)+'.npy'), 'w'), y)



def preprocess_images(raw_data_path, save_data_path, img_scale=0.25, nb_classes=12,
                      img_feature_type='cnn'):

    offset = 2
    x = []
    y = []
    for i in xrange(nb_classes):

        files = os.listdir( os.path.join(raw_data_path, str(i+offset)) )
        for f in files:
            if f.split('.')[1].find('jpg') or f.split('.')[1].find('JPG'):
                ## x.append(os.path.join(raw_data_path, str(i+1), f))

                img = extract_image(os.path.join(raw_data_path, str(i+offset), f),
                                    img_feature_type=img_feature_type,
                                    img_scale=img_scale)
                x.append(img)                
                y.append(i)

    if img_feature_type == 'cnn' or img_feature_type == 'vgg':
        from keras.utils.np_utils import to_categorical
        y = to_categorical(y, nb_classes=nb_classes)

    print np.shape(x), np.shape(y)
    np.save(open(os.path.join(save_data_path,'keras','x_train_img_extra.npy'), 'w'), x)
    np.save(open(os.path.join(save_data_path,'keras','y_train_extra.npy'), 'w'), y)
    return


def extract_image(f, img_feature_type='vgg', img_scale=None):

    # Extract image features
    if img_feature_type is 'cnn':
        img = cv2.imread(f)
        height, width = np.array(img).shape[:2]
        img = cv2.resize(img,(int(width*img_scale), int(height*img_scale)),
                         interpolation = cv2.INTER_CUBIC).astype(np.float32)

        # for vgg but lets use
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        img = img.transpose((2,0,1))

    elif img_feature_type is 'vgg':
        # BGR, vgg
        img_size = (256,256)
        img = cv2.resize(cv2.imread(f), img_size).astype(np.float32)
        crop_size = (224,224)
        img = img[(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
                  ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2,:]

        # for vgg but lets use
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68

        ## if viz:
        ##     print np.shape(img), type(img), type(img[0][0][0])
        ##     # visual check
        ##     cv2.imshow('image',img)
        ##     cv2.waitKey(0)
        ##     cv2.destroyAllWindows()
        ##     sys.exit()
        img = img.transpose((2,0,1))

    elif img_feature_type == 'hog':
        img = cv2.imread(f,0)
        height, width = np.array(img).shape[:2]
        img = cv2.resize(img,(int(width*img_scale), int(height*img_scale)),
        interpolation = cv2.INTER_CUBIC)
        img = hog(img, bin_n=16)

    elif img_feature_type == 'sift':
        img = cv2.imread(f)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray,None)
        img=cv2.drawKeypoints(gray,kp,img)
        print np.shape(img)
        sys.exit()
    else:
        print "Not available"
        sys.exit()

    return img
    

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

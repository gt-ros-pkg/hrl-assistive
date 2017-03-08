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


def preprocess_data(src_pkl, save_data_path, img_scale=0.25, nb_classes=12,
                    img_feature_type='vgg', verbose=False, nFold=None):
    ''' Preprocessing data '''

    save_data_path = os.path.join(save_data_path, 'keras')
    if os.path.isdir(save_data_path) is False:
        os.system('mkdir -p '+save_data_path)

    d = ut.load_pickle(src_pkl)
    if nFold is None:
        nFold = len(d.keys())
    ## print d.keys(), len(d.keys())
    ## nFold = 8
    ## print d.keys()

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
            if img_feature_type in ['cnn', 'vgg', 'cascade']:
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
                img = extract_image(os.path.join(raw_data_path, str(i+offset), f),
                                    img_feature_type=img_feature_type,
                                    img_scale=img_scale)
                x.append(img)                
                y.append(i)

    if img_feature_type in ['cnn', 'vgg', 'cascade']:
        from keras.utils.np_utils import to_categorical
        y = to_categorical(y, nb_classes=nb_classes)

    print np.shape(x), np.shape(y)
    np.save(open(os.path.join(save_data_path,'keras','x_train_img_extra.npy'), 'w'), x)
    np.save(open(os.path.join(save_data_path,'keras','y_train_extra.npy'), 'w'), y)
    return


def extract_image(f, img_feature_type='vgg', img_scale=None, img_ordering='th'):
    '''
    cv2 imread follows th ordering.
    '''

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

        img = img.transpose((2,0,1))

    elif img_feature_type is 'cascade':
        img_size = (224,224) #(256,256)
        crop_size = (224,224)

        img = cv2.resize(cv2.imread(f), img_size)
        rows,cols = np.shape(img)[:2]
        M   = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        img = cv2.warpAffine(img,M,(cols,rows))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceDetectClassifier = cv2.CascadeClassifier("/home/dpark/util/opencv-2.4.11/data/haarcascades/haarcascade_frontalface_default.xml")
        facePoints = faceDetectClassifier.detectMultiScale(gray, 1.3, 5)

        if len(facePoints)==0:
            # Find a head using another way!!
            lower = np.array([0, 18, 40], dtype = "uint8")
            upper = np.array([20, 255, 255], dtype = "uint8")
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            skinMask = cv2.inRange(converted, lower, upper)

            # apply a series of erosions and dilations to the mask
            # using an elliptical kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            skinMask = cv2.erode(skinMask, kernel, iterations = 2)
            skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

            # blur the mask to help remove noise, then apply the
            # mask to the frame
            skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
            skin = cv2.bitwise_and(img, img, mask = skinMask)

            # column: vertical, row: horizontal
            X = []
            for c in xrange(len(skin)):
                for r in xrange(len(skin[c])):
                    if sum(skin[c][r]) == 0: continue
                    X.append([c,r])

            n_clusters = 3
            from sklearn.cluster import KMeans
            clt = KMeans(n_clusters = n_clusters)
            clt.fit(X)

            min_dist = 100000000000000
            min_idx  = []
            for nc in xrange(n_clusters):
                inds = [i for i in xrange(len(clt.labels_)) if clt.labels_[i]==nc ]
                r =0
                c =0
                for i in inds:
                    c += X[i][0]
                    r += X[i][1]
                c /= len(inds)
                r /= len(inds)
                ## c = int(np.mean(X[inds][0]))
                ## r = int(np.mean(X[inds][1]))

                dist = ((len(skin)/2-c)**2 + (len(skin[1])/2-r)**2)
                if min_dist > dist:
                    min_idx = [c,r]
                    min_dist = dist
            c = min_idx[0]
            r = min_idx[1]
        else:        
            r,c,w,h = facePoints[0]
            ## for (r,c,w,h) in facePoints:
            ##     cv2.rectangle(img,(r,c),(r+w,c+h),(255,0,0),2)
            ## skin = img.copy()
            r += w/2
            c += h/2


        ## keypoints = []
        ## keypoints.append( cv2.KeyPoint(r, c, 10 ) )
        ## ## keypoints.append( cv2.KeyPoint(r+w/2, c+h/2, 10 ) )
        ## im_with_keypoints = cv2.drawKeypoints(skin, keypoints, None)            
        if c - crop_size[1]/2 < 0 or c + crop_size[1]/2 > len(img) or\
            r - crop_size[0]/2 < 0 or r + crop_size[0]/2 > len(img[0]):
            # Add replicated border
            img = cv2.copyMakeBorder(img[1:,1:],150,150,150,150,cv2.BORDER_REPLICATE)
            r += 150-1
            c += 150-1
                        
        img = img.astype(np.float32)      
        img = img[c - crop_size[0]/2: c + crop_size[0]/2, r - crop_size[1]/2: r + crop_size[1]/2]

        ## cv2.imshow('image', np.hstack([img.astype(np.uint8), im_with_keypoints]) )
        ## cv2.waitKey(0)
        ## cv2.destroyAllWindows()
        
        rows,cols = np.shape(img)[:2]
        M   = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        img = cv2.warpAffine(img,M,(cols,rows))

        # for vgg but lets use
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
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
    

def hog(img, bin_n=16):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

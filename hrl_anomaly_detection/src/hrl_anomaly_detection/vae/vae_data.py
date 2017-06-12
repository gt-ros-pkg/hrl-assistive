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

# system & utils
import os, sys, copy, random
import numpy as np
import scipy
import hrl_lib.util as ut
from joblib import Parallel, delayed

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
from hrl_execution_monitor import util as autil

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv

# Keras
import h5py 
from keras.models import Sequential, Model
from keras.layers import Merge, Input
from keras.layers import Activation, Dropout, Flatten, Dense, merge, Lambda, LSTM, RepeatVector
from keras.layers.advanced_activations import PReLU
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras import backend as K
from keras import objectives


# visualization
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
random.seed(3334)
np.random.seed(3334)



def gen_data(subject_names, task_name, raw_data_path, processed_data_path, param_dict):
    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    # SVM
    SVM_dict   = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']

    # Adaptation
    ## ADT_dict = param_dict['ADT']

    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']
    
    #------------------------------------------

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)         
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=data_dict['isolationFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])

        d['successData'], d['failureData'], d['success_files'], d['failure_files'], d['kFoldList'] \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        ut.save_pickle(d, crossVal_pkl)

    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    print np.shape(d['successData'])

    d['successData']    = d['successData'][feature_list]
    d['failureData']    = d['failureData'][feature_list]
    print np.shape(d['successData'])


    td = get_ext_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
                      init_param_dict=d['param_dict'])


    # split data
    # HMM-induced vector with LOPO
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(d['kFoldList']):

        # dim x sample x length
        normalTrainData   = copy.deepcopy(d['successData'][:, normalTrainIdx, :])
        abnormalTrainData = copy.deepcopy(d['failureData'][:, abnormalTrainIdx, :])
        normalTestData    = copy.deepcopy(d['successData'][:, normalTestIdx, :]) 
        abnormalTestData  = copy.deepcopy(d['failureData'][:, abnormalTestIdx, :])

        normalTrainData   = np.hstack([normalTrainData, copy.deepcopy(td['successData'])])
        abnormalTrainData = np.hstack([abnormalTrainData, copy.deepcopy(td['failureData'])])
        batch_size = len(normalTrainData[0,0])

        normalTrainData   = np.swapaxes(normalTrainData, 0,1 )
        normalTrainData   = np.swapaxes(normalTrainData, 1,2 )
        abnormalTrainData = np.swapaxes(abnormalTrainData, 0,1 )
        abnormalTrainData = np.swapaxes(abnormalTrainData, 1,2 )

        normalTestData   = np.swapaxes(normalTestData, 0,1 )
        normalTestData   = np.swapaxes(normalTestData, 1,2 )
        abnormalTestData = np.swapaxes(abnormalTestData, 0,1 )
        abnormalTestData = np.swapaxes(abnormalTestData, 1,2 )


        # flatten the data (sample, length, dim)
        trainData = np.vstack([normalTrainData, abnormalTrainData])
        trainData = trainData.reshape(len(trainData)*len(trainData[0]), len(trainData[0,0]))

        testData = normalTestData.reshape(len(normalTestData)*len(normalTestData[0]), len(normalTestData[0,0]))
        print np.shape(trainData), np.shape(testData)

        ## print np.amin(trainData, axis=0)
        ## print np.amax(trainData, axis=0)
        ## sys.exit()

        weights_path = os.path.join(save_data_path,'tmp_weights_'+str(idx)+'.h5')        
        vae = variational_autoencoder(trainData, testData, weights_path, batch_size=batch_size)


        return
    


def get_ext_data(subject_names, task_name, raw_data_path, processed_data_path, param_dict,
                 init_param_dict=None):
    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    
    #------------------------------------------

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    if init_param_dict is None:
        crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
        init_param_dict = d['param_dict']
        

    #------------------------------------------
    
    subjects = ['Andrew', 'Britteney', 'Joshua', 'Jun', 'Kihan', 'Lichard', 'Shingshing', 'Sid', 'Tao']
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/CORL2017/'

    crossVal_pkl = os.path.join(processed_data_path, 'cv_td_'+task_name+'.pkl')
    if os.path.isfile(crossVal_pkl) and data_renew is False and False:
        print "CV data exists and no renew"
        td = ut.load_pickle(crossVal_pkl)
    else:
        # Extract data from designated location
        td = dm.getDataLOPO(subjects, task_name, raw_data_path, save_data_path,\
                            downSampleSize=data_dict['downSampleSize'],\
                            init_param_dict=init_param_dict,\
                            handFeatures=data_dict['isolationFeatures'], \
                            cut_data=data_dict['cut_data'],\
                            data_renew=data_renew, max_time=data_dict['max_time'],
                            pkl_prefix='tgt_', depth=True)

        td['successData'], td['failureData'], td['success_files'], td['failure_files'], td['kFoldList'] \
          = dm.LOPO_data_index(td['successDataList'], td['failureDataList'],\
                               td['successFileList'], td['failureFileList'])

        ut.save_pickle(td, crossVal_pkl)


    #------------------------------------------
    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    print np.shape(td['successData'])

    td['successData']    = td['successData'][feature_list]
    td['failureData']    = td['failureData'][feature_list]
    print np.shape(td['successData'])

    return td





def lstm_test(subject_names, task_name, raw_data_path, processed_data_path, param_dict):
    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    
    #------------------------------------------
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)         
    else:
        sys.exit()

    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    print np.shape(d['successData'])

    d['successData']    = d['successData'][feature_list]
    d['failureData']    = d['failureData'][feature_list]
    print np.shape(d['successData'])

    td = get_ext_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
                      init_param_dict=d['param_dict'])


    # split data
    # HMM-induced vector with LOPO
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(d['kFoldList']):

        # dim x sample x length
        normalTrainData   = copy.deepcopy(d['successData'][:, normalTrainIdx, :])
        abnormalTrainData = copy.deepcopy(d['failureData'][:, abnormalTrainIdx, :])
        normalTestData    = copy.deepcopy(d['successData'][:, normalTestIdx, :]) 
        abnormalTestData  = copy.deepcopy(d['failureData'][:, abnormalTestIdx, :])
        normalTrainData   = np.hstack([normalTrainData, copy.deepcopy(td['successData'])])
        abnormalTrainData = np.hstack([abnormalTrainData, copy.deepcopy(td['failureData'])])
        
        normalTrainData   = np.swapaxes(normalTrainData, 0,1 )
        normalTrainData   = np.swapaxes(normalTrainData, 1,2 )
        abnormalTrainData = np.swapaxes(abnormalTrainData, 0,1 )
        abnormalTrainData = np.swapaxes(abnormalTrainData, 1,2 )

        normalTestData   = np.swapaxes(normalTestData, 0,1 )
        normalTestData   = np.swapaxes(normalTestData, 1,2 )
        abnormalTestData = np.swapaxes(abnormalTestData, 0,1 )
        abnormalTestData = np.swapaxes(abnormalTestData, 1,2 )


        # flatten the data (sample, length, dim)
        trainData = np.vstack([normalTrainData, abnormalTrainData])
        ## trainData = trainData.reshape(len(trainData)*len(trainData[0]), len(trainData[0,0]))
        trainData = [normalTrainData, None]
        testData  = [np.vstack([normalTestData, abnormalTestData]),
                     [0]*len(normalTestData)+[1]*len(abnormalTestData)]
        ## testData = normalTestData.reshape(len(normalTestData)*len(normalTestData[0]), len(normalTestData[0,0]))        

        # Apply time window
        




        weights_path = [os.path.join(save_data_path,'tmp_weights_'+str(idx)+'.h5'),
                        os.path.join(save_data_path,'tmp_weights2_'+str(idx)+'.h5')]
        encoder = lstm_vae(trainData, testData, weights_path, batch_size=5)


        return
    
    
    # flatten data window 1
    



    #def train_vae_classifier() 


def lstm_vae(trainData, testData, weights_file=None, batch_size=140, nb_epoch=50, patience=5):
    """
    Variational AE with LSTM
    x_train is (sample x length x dim)
    x_test is (sample x length x dim)

    Note: it uses offline data.
    """
    x_train = trainData[0]
    y_train = trainData[1]
    x_test = testData[0]
    y_test = testData[1]

    timesteps = len(x_train[0])
    input_dim = len(x_train[0][0])

    h1_dim = input_dim*2
    h2_dim = h1_dim*1
    z_dim  = 2
    
    def sampling(args):
        z_mean, z_log_std = args
        epsilon = K.random_normal(shape=(z_dim,), mean=0., std=.1)
        return z_mean + K.exp(z_log_std) * epsilon
    
    def vae_loss(inputs, x_decoded_mean):
        xent_loss =  K.mean(objectives.binary_crossentropy(inputs, x_decoded_mean), axis=-1)
        kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
        return xent_loss + kl_loss
    
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(h1_dim, return_sequences=True)(inputs)
    encoded = LSTM(h2_dim)(encoded)
    z_mean = Dense(z_dim)(encoded)
    z_log_std = Dense(z_dim)(encoded)
    
    z = Lambda(sampling)([z_mean, z_log_std])
    decoded = Dense(h2_dim)(z)
    decoded = RepeatVector(timesteps)(decoded)
    decoded = LSTM(h1_dim, return_sequences=True)(decoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)
    
    variational_autoencoder = Model(inputs, decoded)
    variational_encoder = Model(inputs, z_mean)
    variational_autoencoder.compile(optimizer='rmsprop', loss=vae_loss)


    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                               verbose=0, mode='auto'),
                ModelCheckpoint(weights_file[0],
                                save_best_only=True,
                                save_weights_only=True,
                                monitor='val_loss'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=20, min_lr=0.00001)]


    if weights_file is not None and os.path.isfile(weights_file[0]) and False:
        variational_autoencoder.load_weights(weights_file[0])
        variational_encoder.load_weights(weights_file[1])
    else:
        variational_autoencoder.fit(x_train, x_train,
                                    shuffle=True,
                                    nb_epoch=nb_epoch,
                                    batch_size=batch_size,
                                    callbacks=callbacks,
                                    validation_data=(x_test, x_test))
        variational_autoencoder.save_weights(weights_file[0])
        variational_encoder.save_weights(weights_file[1])

    # display a 2D plot of classes in the latent space
    x_test_encoded = variational_encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, s=100)
    plt.colorbar()
    plt.show()
        
    return variational_encoder


def variational_autoencoder(x_train, x_test, weights_file=None, batch_size=140):

    # train the VAE on MNIST digits
    ## (x_train, y_train), (x_test, y_test) = mnist.load_data()

    batch_size = batch_size
    original_dim = len(x_train[0])
    latent_dim = 2
    intermediate_dim = len(x_train[0])/2
    nb_epoch = 50
    epsilon_std = 1.0

    print "Dimension info: ", len(x_train[0]), len(x_train[0])/2, 2
    print "Size: ", len(x_train), " batch_size: ", batch_size

    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)


    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  std=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h    = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded    = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)


    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer='rmsprop', loss=vae_loss)


    x_train = x_train.astype('float32') 
    x_test = x_test.astype('float32') 

    if weights_file is not None and os.path.isfile(weights_file):
        vae.load_weights(weights_file)
    else:
        vae.fit(x_train, x_train,
                shuffle=True,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                validation_data=(x_test, x_test))
        vae.save_weights(weights_file)

    # build a model to project inputs on the latent space
    ## encoder = Model(x, z_mean)

    return vae
    
    


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    
    p.add_option('--gen_data', '--gd', action='store_true', dest='gen_data',
                 default=False, help='Generate data.')
    p.add_option('--ext_data', '--ed', action='store_true', dest='extra_data',
                 default=False, help='Add extra data.')
    p.add_option('--preprocess', '--p', action='store_true', dest='preprocessing',
                 default=False, help='Preprocess')
    p.add_option('--lstm_test', '--lt', action='store_true', dest='lstm_test',
                 default=False, help='Generate data.')
    
    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center   = 'kinEEPos'        
    scale       = 1.0
    local_range = 10.0
    nPoints     = 40 #None

    from hrl_anomaly_detection.vae.vae_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    # Mikako - bad camera
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/TCDS2017/'+opt.task+'_data_adaptation2'


    param_dict['data_param']['handFeatures'] = ['unimodal_kinVel',\
                                                'unimodal_kinJntEff_1',\
                                                'unimodal_ftForce_zero',\
                                                'unimodal_ftForce_integ',\
                                                'unimodal_kinEEChange',\
                                                'unimodal_kinDesEEChange',\
                                                'crossmodal_landmarkEEDist', \
                                                'unimodal_audioWristRMS',\
                                                'unimodal_fabricForce',\
                                                'unimodal_landmarkDist',\
                                                'crossmodal_landmarkEEAng']


    if opt.gen_data:
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        gen_data(subjects, opt.task, raw_data_path, save_data_path, param_dict)
        
    elif opt.extra_data:
        
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        get_ext_data(subjects, opt.task, raw_data_path, save_data_path, param_dict)
        

    elif opt.preprocessing:
        src_pkl = os.path.join(save_data_path, 'isol_data.pkl')
        from hrl_execution_monitor import preprocess as pp
        pp.preprocess_data(src_pkl, save_data_path, img_scale=0.25, nb_classes=12,
                            img_feature_type='vgg', nFold=nFold)

    elif opt.lstm_test:
        lstm_test(subjects, opt.task, raw_data_path, save_data_path, param_dict)
        


    print save_data_path

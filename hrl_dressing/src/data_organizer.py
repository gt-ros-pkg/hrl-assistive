# Hidden Markov Model Implementation

import pylab as pyl
import numpy as np
import matplotlib.pyplot as pp

import pickle

import unittest
import ghmmwrapper
import ghmm
import random

import os, os.path

# Define features

def feature_vector(Zt, cat):
    dim = len(Zt)
    vec = np.zeros([dim, 2])
    if cat in [3, 2, 1, 0]:
        for i in range(0, dim):
            vec[i, 0] = random.random() * 1 + 0.0
            vec[i, 1] = random.random() * 1 + 0.0
            #vec[i, 2] = random.random() * 1 + 0.0

    temp = np.dstack([Zt[:, 4], Zt[:, 2]])[0]
    #temp = Zt[:,2:5]

    #if cat == 0:
        #temp = temp[50:800]
        #vec = vec[50:800]
        #temp += vec

    return abs(temp*1000)


if __name__ == '__main__' or __name__ != '__main__':

    data_path = '/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing/'
    exp_list = ['missed/', 'good/', 'caught_fist/', 'caught_other/']
    velocity_list = ['0.1mps/', '0.15mps/']

    Fmat_original = []

## Trials
    subject_list = range(0, 6)
    total_trial = 0
    for i in subject_list:
        path = data_path + 'subject' + str(i) + '/auto_labeled/'
        # path = data_path + 'subject' + str(i) + '/formatted/'
        subject_data = []
        for vel_folders in velocity_list:
            velocity_data = []
            if 'mps' in vel_folders:
                for exp in exp_list:
                    force_data = []
                    force_file_list = os.listdir(path + vel_folders + "/" + exp + '/')
                    for f_ind in xrange(100000):
                        if 'force_profile_'+str(f_ind)+'.pkl' in force_file_list:
                            # print path + vel_folders + "/" + exp + '/' + 'force_profile_'+str(f_ind)+'.pkl'
                            cur_data = pickle.load(open(path + vel_folders + "/" + exp + '/' + 'force_profile_'+str(f_ind)+'.pkl'))
                            force_data.append(feature_vector(cur_data, exp_list.index(exp)))
                        else:
                            break
                    velocity_data.append(force_data)
                subject_data.append(velocity_data)
        Fmat_original.append(subject_data)
    # abc

    '''
    Fmat_original = []
    for i in range(6):
        subject_data = []
        for v in range(2):
            velocity_data = []
            for exp in range(4):
                force_data = []
                for trial in range(random.randint(1,6)):
                    dim = random.randint(1000, 1700)
                    moc_data = np.zeros(dim)
                    moc_data2 = np.zeros(dim)
                    if exp == 0:
                        for j in range(1,  dim):
                            #dif = -j * 1.0/(dim-1)
                            dif = -0.5
                            moc_data[j] = moc_data[j-1] + (random.random() + dif)*10
                            moc_data2[j] = moc_data2[j-1] + (random.random()+dif*0.9)*10
                        for j in range(dim):
                            moc_data[j] += random.random()*0
                            moc_data2[j] += random.random()*0
                    elif exp == 1:
                        for j in range(0,  dim):
                            moc_data[j] = random.random()*0.8 - 15
                            moc_data2[j] = random.random()*5 + 0
                    elif exp == 2:
                        for j in range(0,  dim):
                            moc_data[j] = random.random()*20 - 1
                            moc_data2[j] = random.random()*10 - 10
                    elif exp == 3:
                        for j in range(0,  dim):
                            moc_data[j] = random.random()*15 + 0.5
                            moc_data2[j] = random.random()*0.7 + 9
                    force_data.append(np.dstack([moc_data, moc_data2])[0])
                velocity_data.append(force_data)
            subject_data.append(velocity_data)
        Fmat_original.append(subject_data)
    '''











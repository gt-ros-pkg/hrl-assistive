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
    scalar = 1
    temp = np.dstack([Zt[:,1], (Zt[:,4]) * scalar, (Zt[:, 2]) * scalar])[0]

    return abs(temp)


if __name__ == '__main__' or __name__ != '__main__':

    data_path = '/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing/'
    exp_list = ['missed/', 'good/', 'caught_fist/', 'caught_other/']
    velocity_list = ['0.1mps/', '0.15mps/']

    Fmat_original = []
    Subject_armlength = []

## Trials
    #pp.figure(4)
    #symbols = ['r', 'g', 'b', 'y']
    #vel_symbols = ['+', '-']
    subject_list = range(0, 6)+range(7, 11)
    total_trial = 0
    for i in subject_list:
        path = data_path + 'subject' + str(i) + '/formatted/'
        subject_data = []
        for vel_folders in velocity_list:
            velocity_data = []
            if 'mps' in vel_folders:
                for exp in exp_list:
                    force_data = []
                    force_file_list = os.listdir(path + vel_folders + "/" + exp + '/')
                    for f_ind in xrange(100000):
                        if 'force_profile_'+str(f_ind)+'.pkl' in force_file_list:
                            cur_data = pickle.load(open(path + vel_folders + "/" + exp + '/' + 'force_profile_'+str(f_ind)+'.pkl'))
                            force_data.append(feature_vector(cur_data, exp_list.index(exp)))
                            #pp.plot(range(len(cur_data[:, 4])), abs(cur_data[:, 4]), symbols[exp_list.index(exp)]+vel_symbols[velocity_list.index(vel_folders)])
                            #if velocity_list.index(vel_folders) == 0:
                            #    pp.plot(cur_data[:, 0], abs(cur_data[:, 4]), symbols[exp_list.index(exp)]+vel_symbols[velocity_list.index(vel_folders)])
                        else:
                            break
                    velocity_data.append(force_data)
                subject_data.append(velocity_data)
        Fmat_original.append(subject_data)
        yaml_file = open(data_path + 'subject' + str(i) + '/params.yaml')
        first_line = yaml_file.readline()
        arm_length = float(first_line.split(' ')[1])
        Subject_armlength.append(arm_length)
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











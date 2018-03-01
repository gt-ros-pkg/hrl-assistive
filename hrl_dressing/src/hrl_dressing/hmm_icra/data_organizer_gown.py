#!/usr/bin/env python
# Hidden Markov Model Implementation

import pylab as pyl
import numpy as np
import matplotlib.pyplot as pp
import rospkg

import pickle

import random

import math

import os, os.path

# from dtw import dtw



def get_position_trunc_index(position, positions):
    for i in xrange(len(positions)):
        if positions[i] > position:
            return i
    return len(positions)

def get_maxforce_percent_index(force_threshold, forces):
    for i in xrange(len(forces)):
        magnitude = 0
        for j in xrange(len(forces[i])):
            magnitude += (forces[i][j] * forces[i][j])
        if math.sqrt(magnitude) > force_threshold:
            #print 'percentage of data: ', i * 1.0 / len(forces)
            if i == 0:
                i = 1
            return i
    return len(forces)


def warp_slow_to_fast(data):
    interpolated_data = []
    for i in xrange(len(data)-1):
        interpolated_data.append(data[i])
        interpolated_data.append(data[i] + 0.5*(data[i+1]-data[i]))
    interpolated_data.append(data[len(data)-1])
    interpolated_data = np.array(interpolated_data)
    warped_data = []
    for i in xrange(len(interpolated_data)):
        if i%3 == 0:
            warped_data.append(interpolated_data[i])
    return np.array(warped_data)

def feature_vector(Zt):
    orig = Zt[0,1]
    temp = np.dstack([Zt[:,1], (Zt[:,2])*1, (Zt[:, 4])*1, -(Zt[:, 3])*1])[0]
    return temp

def pkl_data(fname):
    cur_data = pickle.load(open(fname))
    return feature_vector(cur_data)

def random_noise():
    rd = random.random()
    if rd < 0.18:
        return -0.01 + random.random()*0.002
    elif rd >= 1-0.18:
        return 0.01 + random.random()*0.002
    return 0

def txt_data(fname):
    cur_file = open(fname)
    cur_data = [[]]
    line = cur_file.readline()
    orig = -100
    expid = -1
    if fname[-6:-4].isdigit():
        expid = int(fname[-6:-4])
    elif fname[-5:-4].isdigit():
        expid = int(fname[-5:-4])

    #if expid < 10 or (expid > 40 and expid < 61) or expid > 90:
    #    return []

    while (line != ''):
        line = line.split()
        rd = random.random()

        # immitate random noise from real-world data
        # not used for now
        randmove=random_noise()*0
        randgrav = random_noise()*0
        if orig == -100:
            orig = float(line[1])

        if len(cur_data[0]) == 0:
            cur_data[0] = [abs(float(line[1])-orig), (float(line[2])+randmove)*1, (float(line[3])+randgrav)*1, float(line[4])]
        else:
            cur_data = np.vstack([cur_data, [abs(float(line[1])-orig), (float(line[2])+randmove)*1, (float(line[3])+randgrav)*1, float(line[4])]])
        line = cur_file.readline()

    return cur_data

if __name__ == '__main__' or __name__ != '__main__':

    data_path = '/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing/'
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hrl_dressing')
    exp_list = ['missed/', 'good/', 'caught/']
    vel_list = ['0.1mps/', '0.15mps/']
    test_subjects = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

    test_path_one = 'test_subject/subject0_onetrial/'
    test_path_one = 'train_trial_subject0_onlywind/'
    #test_path_one = 'train_onetrial_subject0/'
    train_path_list = []
    for i in test_subjects:
        train_path_list.append('subject'+str(i)+'/formatted_three/')

    # print train_path_list
    # variables to be imported
    Fmat_original = []

    Maximal_initial_position = 0

    avg_miss_end = []
    avg_caught_end = []
    avg_good_end = []

    i = 0
    for exp in exp_list:
        train_datas = []
        exp_data = []
        for train_data_folder in train_path_list:
            train_data = []
            for vel in vel_list:
                # pp.figure(i+10)
                train_file_path = data_path + train_data_folder + vel + exp


                for datafile in os.listdir(train_file_path):
                    # print datafile
                    if '.pkl' in datafile:
                        td = pkl_data(train_file_path+datafile)
                        # print 'loaded data from ', train_file_path+datafile
                        orig = td[0, 0]
                        for p in range(len(td[:, 0])):
                            td[p, 0] -= orig
                        # print td
                    elif '.txt' in datafile:
                        td = txt_data(train_file_path+datafile)
                        if len(td) == 0:
                            continue
                        if 'f' not in datafile:
                            td = warp_slow_to_fast(td)
                        #else:
                        #    continue

                    else:
                        pass

                    train_data.append(td)
                    # if 'miss' in exp:
                    #     pp.plot(td[:, 2], 'b', alpha=0.8)
                    # if 'good' in exp:
                    #     pp.plot(td[:, 2], 'b', alpha=0.8)
                    # if 'caught' in exp:
                    #     pp.plot(td[:, 2], 'b', alpha=0.8)
            train_datas.append(train_data)
            # print len(train_datas)


        subject_list = [15]
        test_data = []
        test_datas = []
        test_folder = pkg_path + '/data/pr2_test_ft_data/'
        if exp == 'missed/':
            testdata = pkl_data(test_folder + 'ft_sleeve_1.pkl')
        elif exp == 'good/':
            testdata = pkl_data(test_folder + 'ft_sleeve_0.pkl')
        elif exp == 'caught/':
            testdata = pkl_data(test_folder + 'ft_sleeve_2.pkl')
        # print 'testdata', testdata
        test_data.append(testdata)
        test_datas.append(test_data)

        exp_data.append(train_datas)
        exp_data.append(test_datas)
        Fmat_original.append(exp_data)


# Hidden Markov Model Implementation

import pylab as pyl
import numpy as np
import matplotlib.pyplot as pp

import pickle

import unittest
import ghmmwrapper
import ghmm
import random

import math

import os, os.path

# Define features

max_seq_magnitude = []

def feature_vector(Zt, cat):
    dim = len(Zt)
    vec = np.zeros([dim, 2])
    if cat in [3, 2,1,0]:
        for i in range(0, dim):
            vec[i, 0] = random.random() * 1 + 0.0
            vec[i, 1] = random.random() * 1 + 0.0
            #vec[i, 2] = random.random() * 1 + 0.0

    scalar = 1
    max_magnitude = -1
    for i in range(dim):
        mag = (Zt[i, 2] * Zt[i, 2]) + (Zt[i, 4] * Zt[i, 4])
        mag = math.sqrt(mag)
        if mag > max_magnitude:
            max_magnitude = mag
    max_seq_magnitude.append(max_magnitude)

    #log
    #temp = np.dstack([Zt[:,1], np.log(abs(Zt[:,4])*100+1)*100 * scalar, np.log(abs(Zt[:, 2])*100+1)*100 * scalar])[0]
    temp = np.dstack([Zt[:,1], (Zt[:,4]) * scalar, (Zt[:, 2]) * scalar])[0]
    #temp = Zt[:,2:5]

    #if cat == 0:
        #temp = temp[200:]
        #vec = vec[200:]
        #temp += vec

    #return np.log(abs(temp*100)+10)*100
    return temp

def txt_data(fname):
    cur_file = open(fname)
    cur_data = [[]]
    line = cur_file.readline()
    orig = 0
    while (line != ''):
        line = line.split()
        randmove = (random.random() - 0.5) * 0.0
        randgrav = (random.random() - 0.5) * 0.0
        if len(cur_data[0]) == 0:
            orig = float(line[1])
            cur_data[0] = [float(line[1])-orig, (float(line[2])+randmove)*1, (float(line[3])+randgrav)*1]
        else:
            cur_data = np.vstack([cur_data, [abs(float(line[1])-orig), (float(line[2])+randmove)*1, (float(line[3])+randgrav)*1]])
        line = cur_file.readline()
    return cur_data

if __name__ == '__main__' or __name__ != '__main__':

    data_path = '/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing/'
    #data_path = '/Users/yuwenhao/robotics/data/dressing_fake/'
    #exp_list = ['missed/', 'good/', 'caught_fist/', 'caught_other/']
    exp_list = ['missed/', 'good/', 'caught/']
    #exp_list = ['caught/', 'good/']
    velocity_list = ['0.1mps/', '0.15mps/']
    #velocity_list = ['0.1mps/']

    # variables to be imported
    Fmat_original = []
    Subject_armlength = []
    Maximal_initial_position = -1000

    avg_miss_end = []
    avg_caught_end = []
    avg_good_end = []

## Trials
    #pp.figure(4)
    #symbols = ['r', 'g', 'b', 'y']
    #vel_symbols = ['+', '-']
    subject_list = range(0, 6)+range(7, 13)
    subject_list = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    total_trial = 0
    for i in subject_list:
        path = data_path + 'subject' + str(i) + '/formatted_three/'
        subject_data = []
        for vel_folders in velocity_list:
            velocity_data = []
            for exp in exp_list:
                force_data = []
                force_file_list = os.listdir(path + vel_folders + "/" + exp + '/')
                for force_files in force_file_list:
                    if '.pkl' in force_files:
                        cur_data = pickle.load(open(path + vel_folders + "/" + exp + '/' + force_files))
                        force_data.append(feature_vector(cur_data, exp_list.index(exp)))
                    if '.txt' in force_files:
                        force_data.append(txt_data(path + vel_folders + "/" + exp + '/' + force_files))
                velocity_data.append(force_data)
            subject_data.append(velocity_data)
        Fmat_original.append(subject_data)
        yaml_file = open(data_path + 'subject' + str(i) + '/params.yaml')
        first_line = yaml_file.readline()
        arm_length = float(first_line.split(' ')[1])
        #arm_length = 44
        Subject_armlength.append(arm_length)

        # process position data to be absolute position
        for v in range(len(Fmat_original[-1])):
            for c in range(len(Fmat_original[-1][v])):
                for t in range(len(Fmat_original[-1][v][c])):
                    for i in range(len(Fmat_original[-1][v][c][t])):
                        Fmat_original[-1][v][c][t][i][0] = Fmat_original[-1][v][c][t][i][0] + 0.44 - arm_length/100.0
                    if Fmat_original[-1][v][c][t][0][0] > Maximal_initial_position:
                        Maximal_initial_position = Fmat_original[-1][v][c][t][0][0]
                    if c == 0:
                        avg_miss_end.append(Fmat_original[-1][v][c][t][-1][0])
                    elif c == 1:
                        avg_good_end.append(Fmat_original[-1][v][c][t][-1][0])
                    elif c == 2:
                        avg_caught_end.append(Fmat_original[-1][v][c][t][-1][0])
    '''
    print 'avg miss end: ', np.mean(np.array(avg_miss_end))
    print 'avg good end: ', np.mean(np.array(avg_good_end))
    print 'avg caught end: ', np.mean(np.array(avg_caught_end))
    print 'std miss end: ', np.std(np.array(avg_miss_end))
    print 'std good end: ', np.std(np.array(avg_good_end))
    print 'std caught end: ', np.std(np.array(avg_caught_end))
    print 'max magnitude: ', max(max_seq_magnitude)
    print 'min max magnitude: ', min(max_seq_magnitude)
    '''
    print 'maximal initial position: ', Maximal_initial_position

    #pp.show()
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
                            moc_data2[j] = moc_data2[j-1] + (random.random()+dif)*10
                        for j in range(dim):
                            moc_data[j] += random.random()*0
                            moc_data2[j] += random.random()*0
                            moc_data[j] = random.random()*0
                            moc_data2[j] = random.random()*0
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











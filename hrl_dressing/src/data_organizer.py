# Hidden Markov Model Implementation

import pylab as pyl
import numpy as np
import matplotlib.pyplot as pp
#from enthought.mayavi import mlab

import scipy as scp
import scipy.ndimage as ni

import roslib; roslib.load_manifest('sandbox_tapo_darpa_m3')
import rospy
#import hrl_lib.mayavi2_util as mu
import hrl_lib.viz as hv
import hrl_lib.util as ut
import hrl_lib.matplotlib_util as mpu
import pickle

import unittest
import ghmm
import ghmmwrapper
import random

import os, os.path

# Define features

def feature_vector(Zt,i):
	
	return abs(Zt[:,2])

  
if __name__ == '__main__' or __name__ != '__main__':

    data_path = '/home/tapo/svn/robot1_data/usr/tapo/data/dressing/'
    exp_list = ['missed/', 'good/', 'high/', 'caught/']    
    exps = 5

    ta = [0.0]*10000
        
## Trials
    temp_num = 0
    for i in range(np.size(exp_list)):
        path = data_path + exp_list[i]
        for num_file in range(1, exps+1):
            ta[exps*i + num_file - 1] = ut.load_pickle(path + 'force_profile_' + np.str(num_file) + '.pkl')
        temp_num = exps*np.size(exp_list)

    Fmat_original = [0.0]*temp_num

## Creating Feature Vector

    idx = 0
    while (idx < temp_num):
        Fmat_original[idx] = feature_vector(ta[idx],0)
        idx = idx + 1
    


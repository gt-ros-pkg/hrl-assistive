#!/usr/local/bin/python

import sys, os
import numpy as np, math
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import rospy
import inspect
import warnings
import random

# Util
import hrl_lib.util as ut

# Matplot
import matplotlib.pyplot as plt
import matplotlib.pyplot as pp
import hrl_lib.matplotlib_util as mpu

# Sklearn
from sklearn import cross_validation

TOL=0.0001

class traj_data():
    def __init__(self):

        pth       = os.environ['HRLBASEPATH']+'/src/projects/modeling_forces/handheld_hook/'
        blocked_thresh_dict = ut.load_pickle(pth+'blocked_thresh_dict.pkl') # ['mean_charlie', 'mean_known_mech']

        # Get data
        self.semantic = blocked_thresh_dict['mean_charlie'] # each category has (n_std, mn, std)  <= force profiles
        self.second_time = blocked_thresh_dict['mean_known_mech'] # (Ms(mn_mn, var_mn, mn_std, var_std), n_std)=(tuple(4),float)        

        self.max_force = 0.0
        
        pass

    def find_nearest(self, array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]
    
    def discrete_profile(self, plot=False):

        
        # Get max force
        max_force = 0.0        
        for key in self.semantic.keys():
            if max_force < max(self.semantic[key][1]):
                max_force = max(self.semantic[key][1])

        self.max_force = np.ceil(max_force)

        # Discrete Force list
        discrete_table = np.arange(0.0, self.max_force+0.000001, 0.5)
        
        # Discretize it
        test_dict = {}
        for key in self.semantic.keys():

            # Discrete Force profile
            discrete_force = np.zeros(self.semantic[key][1].shape)
            for i,f in enumerate(self.semantic[key][1]):
                discrete_force[i] = self.find_nearest(discrete_table,f)

            test_dict[key] = discrete_force

        # plot
        if plot:
            mpu.set_figure_size(10, 7.0)
            f = pp.figure()
            f.set_facecolor('w')

            for key in self.semantic.keys():
                pp.plot(self.semantic[key][1], 'r-')
                pp.plot(test_dict[key],'b-')
                
            
            pp.ylim(-3., 16.)
            pp.xlim(0., 34.)
            f.subplots_adjust(bottom=.15, top=.99, right=.99)
            ## mpu.legend(loc='lower left')
            ## pp.savefig('collision_detection_hsi_kitchen_pr2.pdf')
            pp.show()

        return test_dict

        
    
    def test(self):

        print self.semantic.keys()
        print self.semantic['kitchen_cabinet_pr2'][1]
        print max(self.semantic['kitchen_cabinet_pr2'][1])

        # With respect to all semantic data, convert those into discrete action-observation

        
        
        return


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()
    
    td        = traj_data()
    test_dict = td.discrete_profile(False)

    kf = cross_validation.KFold(len(test_dict.keys()), n_folds = 2, shuffle=True)
    for train_index, test_index in kf:
        print train_index, test_index

    
    ## td.sampling_histories()

    ## data_path = os.getcwd()    
    ## td.test()

    

    

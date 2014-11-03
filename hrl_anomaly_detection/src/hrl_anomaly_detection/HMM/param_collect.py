#!/usr/local/bin/python

import sys, os, glob
import numpy as np, math
import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy
import inspect
import warnings
import random

# Util
import hrl_lib.util as ut

if __name__ == '__main__':


    data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune'
    output_file = open('/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune/'+'performance.txt', "w")

    
    file_list = glob.glob(data_path+'/*.pkl')
    for pkl_file in file_list:
        try:
            data = ut.load_pickle(pkl_file)
        except:
            print "check file: "+pkl_file
            print "failed to load pickle, corrupted? ..."
            failure_file = os.path.join(data_path,'failure_'+pkl_file+'.txt')
            touch(failure_file)
            ## os.system('rm *') 
            sys.exit()
        if data == None:
            print "failed to load pickle, none there? ..."
            failure_file = os.path.join(data_path,'failure_'+pkl_file+'.txt')
            touch(failure_file)            
            sys.exit()

        
        mean_list = data['mean']
        std_list = data['std']
        params_list = data['params']

        fObsrvResol_list = []
        nState_list = []            
        nCurrentStep_list = []            
        step_size_list_list = []            
        B_list = []
        
        for param in params_list:

            fObsrvResol_list.append(param['fObsrvResol'])
            nState_list.append(param['nState'])
            B_list.append(param['B'])
            ## nCurrentStep_list.append(param['nCurrentStep'])
            ## step_size_list_list.append(param['step_size_list'])

        for i in xrange(len(mean_list)):
            ## string =  "%f; %f; %f; %f; %s \n " % (mean_list[i], std_list[i], fObsrvResol_list[i], nState_list[i], str(step_size_list_list[i]))
            ## string =  "%f; %f; %f; %f; %f; \n " % (mean_list[i], std_list[i], fObsrvResol_list[i], nState_list[i], nCurrentStep_list[i])
            ## string =  "%f; %f; %f; %f; \n " % (mean_list[i], std_list[i], fObsrvResol_list[i], nState_list[i])
            ## string =  "%f; %f; %s \n " % (mean_list[i], std_list[i], str(step_size_list_list[i]))
            string =  "%f; %f; %f; %f; %s \n " % (mean_list[i], std_list[i], fObsrvResol_list[i], nState_list[i], str(B_list[i]))

            output_file.write("%s" % string)
    output_file.close()
            

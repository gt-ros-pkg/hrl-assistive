#!/usr/local/bin/python                                                                                          

# System library
import sys
import os
import time
import math
import numpy as np
import cPickle as pk
#import scipy.io as sio
#import json
import klepto

# ROS library
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import hrl_lib.util as ut

import arm_trajectories as at
import mekabot.hrl_robot
import hrl_hokuyo.hokuyo_scan


def getAllFiles(dirName):

    lFile = []
    for root, dirs, files in os.walk(dirName):

        if root.find('.svn') >= 0: continue
                
        for sub_dir in dirs:
            if sub_dir.find('.svn') >= 0: continue
            lFile = lFile + getAllFiles(sub_dir)

        for sub_file in files:
            lFile.append(os.path.join(root,sub_file))
                    
    return lFile

def removeNaN(list_data):
    temp = np.array(list_data)
    temp_flat = temp.flatten()
    
    for i in xrange(temp_flat.shape[0]):
        if math.isnan(temp_flat[i]):
            temp_flat[i] = 0.0
            
    return temp_flat.reshape(temp.shape).tolist()
    

if __name__ == '__main__':

    dirName='/home/mycroft/dpark_test/hrl_anomaly_detection/matlab/data/'
    dirName='/home/mycroft/svn/robot1_data/usr/advait/ram_www'
    dirName='/home/mycroft/svn/robot1/src/projects/modeling_forces/'
    
    lFile = getAllFiles(dirName)

    for f in lFile:

        fileName, fileExtension = os.path.splitext(f)
        strPath, strFile = os.path.split(f)

        ## if fileExtension.find('py')>=0 and fileName.find('klepto')>=0:
        ##     os.remove(f)
        ##     continue
        
        if fileExtension.find('pkl')>=0 and fileName.find('_new')<0:
            new_pkl_fileName = fileName+'_new.pkl'
            mat_fileName     = fileName+'.mat'

            if os.path.isfile(mat_fileName): continue
            
            if new_pkl_fileName not in lFile:
                try:
                    pkl_data = ut.load_pickle(f)
                except:
                    print "Failure: ", f
                    continue
                
                data = {}
                count = 0 
                               
                for key in pkl_data.keys():
                    if isinstance(pkl_data[key], (float,str,np.float64,int,tuple)):
                        data[key] = pkl_data[key]
                    elif isinstance(pkl_data[key], np.ndarray):
                        data[key] = removeNaN(pkl_data[key].tolist())                                            
                    elif isinstance(pkl_data[key], np.core.defmatrix.matrix):
                        data[key] = pkl_data[key].tolist()                        
                    elif isinstance(pkl_data[key], (at.JointTrajectory)):
                        data[key+'_JT_'+'time_list']    = pkl_data[key].time_list
                        data[key+'_JT_'+'q_list']       = pkl_data[key].q_list
                        data[key+'_JT_'+'qdot_list']    = pkl_data[key].qdot_list
                        data[key+'_JT_'+'qdotdot_list'] = pkl_data[key].qdotdot_list                        
                    elif isinstance(pkl_data[key], (at.PlanarTrajectory)):
                        data[key+'_PT_'+'time_list'] = pkl_data[key].time_list
                        data[key+'_PT_'+'x_list']    = pkl_data[key].x_list
                        data[key+'_PT_'+'y_list']    = pkl_data[key].y_list
                        data[key+'_PT_'+'a_list']    = pkl_data[key].a_list
                    elif isinstance(pkl_data[key], (at.CartesianTajectory)):
                        data[key+'_CT_'+'time_list'] = pkl_data[key].time_list
                        data[key+'_CT_'+'p_list']    = pkl_data[key].p_list
                        data[key+'_CT_'+'v_list']    = pkl_data[key].v_list
                    elif isinstance(pkl_data[key], (at.ForceTrajectory)):
                        data[key+'_FT_'+'time_list'] = pkl_data[key].time_list
                        data[key+'_FT_'+'f_list']    = pkl_data[key].f_list
                    elif isinstance(pkl_data[key], dict):
                        data[key] = pkl_data[key]                        
                    elif isinstance(pkl_data[key], (mekabot.hrl_robot.MekaArmSettings)):
                        data[key+'_MekaArmSettings_'+'stiffness_scale'] = pkl_data[key].stiffness_scale
                        data[key+'_MekaArmSettings_'+'stiffness_list']  = pkl_data[key].stiffness_list
                        data[key+'_MekaArmSettings_'+'control_mode']    = pkl_data[key].control_mode
                    elif isinstance(pkl_data[key], (hrl_hokuyo.hokuyo_scan.HokuyoScan)):
                        data[key+'_HokuyoScan_'+'hokuyo_type'] = pkl_data[key].hokuyo_type
                        data[key+'_HokuyoScan_'+'angular_res'] = pkl_data[key].angular_res
                        data[key+'_HokuyoScan_'+'max_range'] = pkl_data[key].max_range
                        data[key+'_HokuyoScan_'+'min_range'] = pkl_data[key].min_range
                        data[key+'_HokuyoScan_'+'start_angle'] = pkl_data[key].start_angle
                        data[key+'_HokuyoScan_'+'end_angle'] = pkl_data[key].end_angle
                        data[key+'_HokuyoScan_'+'n_points'] = pkl_data[key].n_points
                        data[key+'_HokuyoScan_'+'ranges'] = pkl_data[key].ranges
                        data[key+'_HokuyoScan_'+'intensities'] = pkl_data[key].intensities
                        data[key+'_HokuyoScan_'+'angles'] = pkl_data[key].angles.tolist()                        
                    elif isinstance(pkl_data[key], (list)):

                        if not isinstance(pkl_data[key][0], (np.float64,float,list)):
                        
                            sub_data_list = []

                            for x in pkl_data[key]:
                                try:
                                    sub_data = {}
                                    sub_data[key+'_HokuyoScan_'+'hokuyo_type'] = x.hokuyo_type
                                    sub_data[key+'_HokuyoScan_'+'angular_res'] = x.angular_res
                                    sub_data[key+'_HokuyoScan_'+'max_range'] = x.max_range
                                    sub_data[key+'_HokuyoScan_'+'min_range'] = x.min_range
                                    sub_data[key+'_HokuyoScan_'+'start_angle'] = x.start_angle
                                    sub_data[key+'_HokuyoScan_'+'end_angle'] = x.end_angle
                                    sub_data[key+'_HokuyoScan_'+'n_points'] = x.n_points
                                    sub_data[key+'_HokuyoScan_'+'ranges'] = removeNaN(x.ranges.tolist())
                                    sub_data[key+'_HokuyoScan_'+'intensities'] = removeNaN(x.intensities.tolist())
                                    sub_data[key+'_HokuyoScan_'+'angles'] = removeNaN(x.angles.tolist())
                                    sub_data_list.append(sub_data)
                                except:
                                    print "its not hokuyo"
                                    print type(x)
                                    print x
                                    sys.exit()
                            data[key] = sub_data_list
                                    
                        else:
                            data[key] = removeNaN(pkl_data[key])
                                
                    else:                        
                        print f
                        print "No available type! ", key, type(pkl_data[key])
                        sys.exit()


                        #print data    
                cache = klepto.archives.file_archive(fileName+'_klepto.py', data, serialized=False)
                ## cache = klepto.archives.file_archive('data_klepto.py', data, serialized=False)
                cache.dump()

                ## sio.savemat(mat_fileName,data)

                ## try:
                ##     p = open(fileName+'.json', 'w')
                ## except IOError:
                ##     print "file open error: ", fileName
                ##     break
                ## p.write(json.dumps(data))                
                ## p.close()
                


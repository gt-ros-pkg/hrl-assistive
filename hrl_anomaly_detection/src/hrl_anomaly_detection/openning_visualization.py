#!/usr/local/bin/python                                                                                          

# System library
import sys
import os
import time
import numpy as np

# ROS library
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import hrl_lib.util as ut
import matplotlib.pyplot as plt
import pkl_converter as pk

def plot_all_angle_force(dirName, human=True):
    
    # Plot results
    plt.figure()

    ax = plt.subplot(111, aspect='equal')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    lFile = pk.getAllFiles(dirName)
    for f in lFile:
        strPath, strFile = os.path.split(f)
        if strFile.find('_new.pkl')>=0:

            try:
                data = ut.load_pickle(f)   
            except:
                continue            

            if human:
                if strFile.find('close')>=0:
                    plt.plot(data['mechanism_x']*180.0/np.pi, data['force_tan_list'], "r-")
                else:
                    plt.plot(data['mechanism_x']*180.0/np.pi, data['force_tan_list'], "b-")
            else:
                print data.keys()                
                plt.plot(data['config_list']*180.0/np.pi, data['ftan_list'], "b-")
                   
    plt.show()

# For single file
def plot_angle_force(fileName, human=True):
    
    # Plot results
    plt.figure()

    ax = plt.subplot(111, aspect='equal')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    data = ut.load_pickle(fileName)   

    if human:
        if strFile.find('close')>=0:
            plt.plot(data['mechanism_x']*180.0/np.pi, data['force_tan_list'], "r-")
        else:
            plt.plot(data['mechanism_x']*180.0/np.pi, data['force_tan_list'], "b-")
    else:
        print data.keys()           

        plt.plot(data['ftan_list'], "b-")
                   
    plt.show()



if __name__ == '__main__':

    #dirName='/home/dpark/svn/robot1_data/usr/advait/ram_www/aggregated_pkls_April9_8pm/tests/HSI_kitchen_cabinet_right_charlie'
    ## plot_all_angle_force(dirName, False)

    
    ## dirName='/home/dpark/svn/robot1_data/usr/advait/ram_www/data_from_robot_trials/robot_trials/kitchen_cabinet_locked/pr2_pull_2010Dec12_005340.pkl'    
    ## plot_angle_force(dirName, False)

    ## dirName='/home/dpark/svn/robot1_data/usr/advait/ram_www/data_from_robot_trials/robot_trials/perfect_perception/kitchen_cabinet_collision_box_cody_new.pkl'
    ## plot_angle_force(dirName, False)

    fileName='/home/dpark/svn/robot1_data/usr/advait/ram_www/data_from_robot_trials/robot_trials/hsi_kitchen_collision_box/pull_trajectories_kitchen_cabinet_2010Dec10_060454.pkl'

    import arm_trajectories as at

    data = ut.load_pickle(fileName)   

    
    

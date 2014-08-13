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
## import pkl_converter as pk
import common as co

def plot_all_angle_force(dirName, human=True):
    
    # Plot results
    plt.figure()

    ax = plt.subplot(111, aspect='equal')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    lFile = co.getAllFiles(dirName)
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
def plot_angle_force(fileName, x_name=None, y_name=None,  human=True):
    
    # Plot results
    plt.figure()

    ax = plt.subplot(111)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    data = ut.load_pickle(fileName)   

    if human:
        if strFile.find('close')>=0:
            plt.plot(data['mechanism_x']*180.0/np.pi, data['force_tan_list'], "r-")
        else:
            plt.plot(data['mechanism_x']*180.0/np.pi, data['force_tan_list'], "b-")
    else:

        if x_name == None:
            plt.plot(data[y_name], "b-")
        else:
            x_data = np.array(data[x_name])
            if x_name.find('ang') >= 0 or x_name.find('config') >= 0:
                x_data = x_data * 180.0/np.pi
                ax.set_xlim(0,35)
            else:
                ax.set_xlim(np.min(x_data),np.max(x_data))
                            
            plt.plot(x_data,data[y_name], "b-")
            ax.set_xlabel(x_name)

        ax.set_ylim((min(data[y_name]),max(data[y_name])))
        ax.set_ylabel(y_name)
            
                   
    plt.show()



if __name__ == '__main__':

    ## Human openning
    #dirName='/home/dpark/svn/robot1_data/usr/advait/ram_www/aggregated_pkls_April9_8pm/tests/HSI_kitchen_cabinet_right_charlie'
    ## plot_all_angle_force(dirName, False)

    ## Robot openning with anormal situation
    ## dirName='/home/dpark/svn/robot1_data/usr/advait/ram_www/data_from_robot_trials/robot_trials/kitchen_cabinet_locked/pr2_pull_2010Dec12_005340.pkl'    
    ## plot_angle_force(dirName, y_name='ftan_list' ,human=False)

    ## dirName='/home/dpark/svn/robot1_data/usr/advait/ram_www/data_from_robot_trials/robot_trials/perfect_perception/kitchen_cabinet_collision_box_cody_new.pkl'
    ## plot_angle_force(dirName, False)

    dirName = fileName='/home/dpark/svn/robot1_data/usr/advait/ram_www/data_from_robot_trials/robot_trials/hsi_kitchen_collision_pr2/pr2_pull_2010Dec10_071602_new.pkl'
    dirName = fileName='/home/dpark/svn/robot1/src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/simulate_perception/ikea_cabinet_known_rad_pr2_new.pkl'

    
    import arm_trajectories as at
    data = ut.load_pickle(fileName)   

    print data.keys()
    
    ## plot_angle_force(dirName, x_name="online_ang", y_name='online_ftan' ,human=False)
    ## plot_angle_force(dirName, x_name="config_list", y_name='ftan_list' ,human=False)
    plot_angle_force(dirName, x_name="config", y_name='vec_list' ,human=False)
    

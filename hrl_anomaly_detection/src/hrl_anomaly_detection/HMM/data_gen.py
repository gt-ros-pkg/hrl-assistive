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


TOL=0.0001

class traj_data():
    def __init__(self, data_path):

        self.press = None
        self.traj = None
        self.all_trajs = []
        self.data_path = data_path.rstrip('/')
        self.pkl_file = os.path.join(data_path,'mouse_demo.pkl')

        self.resol = 0.1
        pass

    def get_traj_pkl(self, pkl_file):
        data = ut.load_pickle(pkl_file)
        self.all_trajs = data['all_trajs']
        self.aStart = data['start']
        self.aGoal = data['goal']        
        self.all_discret_trajs = self.discretize_traj(self.all_trajs)
        return data

    def discretize_traj(self, trajs):

        all_discret_trajs = []
        for traj in trajs:
            all_discret_trajs.append(np.around(traj, 1))

        return all_discret_trajs
    
    def get_mouse_traj(self, aStart, aGoal):
        
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111, aspect='equal')

        self.aStart = aStart
        self.aGoal  = aGoal
        
        self.ax.scatter([aStart[0],aGoal[0]],[aStart[1],aGoal[1]], marker='*',s=20,c='r')
        self.ax.set_xlim([-2.0,2.0])
        self.ax.set_ylim([-2.0,2.0])

        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        plt.show()

    def save_mouse_traj(self, pkl_file):

        # save pkl     
        data = {}
        data['all_trajs'] = self.all_trajs
        data['start'] = self.aStart
        data['goal'] = self.aGoal
        ut.save_pickle(data,pkl_file)
        print "Raw Data Pkl Saved"            
        
    def on_press(self, event):

        if event.button == 1:
            'on left button press we will see if the mouse is over us and store some data'
            if event.xdata != None and event.ydata != None:
                self.press = event.xdata, event.ydata
                self.traj = np.array([[event.xdata], [event.ydata]])

        elif event.button == 3:
            self.save_mouse_traj(self.pkl_file)

    def on_release(self, event):

        if event.button == 1:        
            'on release we reset the press data'
            self.press = None
            self.ax.plot(self.traj[0,:], self.traj[1,:], 'r-')
            ## print self.traj[0,:], self.traj[1,:]
            self.fig.canvas.draw()

            self.all_trajs.append(self.traj)
        

    def on_motion(self, event):

        if event.button == 1:        
            if self.press is None: return
            if event.xdata != None and event.ydata != None:
                self.traj = np.hstack([self.traj, np.array([[event.xdata], [event.ydata]])])



if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()

    opt, args = p.parse_args()

    data_path = os.getcwd()
    aStart = np.array([-1.0,-1.0])
    aGoal  = np.array([1.0,1.0])
    
    td = traj_data(data_path)
    td.get_mouse_traj(aStart, aGoal)

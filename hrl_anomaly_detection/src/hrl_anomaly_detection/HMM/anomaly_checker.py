#!/usr/local/bin/python                                                                                          

# System library
import sys
import os
import time
import math
import numpy as np
import glob
import time

# ROS library
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import hrl_lib.util as ut
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec

#
import hrl_lib.circular_buffer as cb
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl


class anomaly_checker():

    def __init__(self, ml, nDim=1, fXInterval=1.0, fXMax=90.0):

        # Object
        self.ml = ml

        # Variables
        self.nFutureStep = self.ml.nFutureStep
        self.nMaxBuf     = self.ml.nFutureStep
        self.nDim        = nDim        
        self.fXInterval  = fXInterval
        self.fXMax       = fXMax
        self.aXRange     = np.arange(0.0,fXMax,self.fXInterval)
        self.fXTOL       = 1.0e-1
        self.fAnomaly    = 8.0
        
        # N-buffers
        self.buf_dict = {}
        for i in xrange(self.nMaxBuf):
            self.buf_dict['mu_'+str(i)] = cb.CircularBuffer(i+1, (nDim,))       
            self.buf_dict['sig_'+str(i)] = cb.CircularBuffer(i+1, (nDim,))       

        # x buffer
        ## self.x_buf = cb.CircularBuffer(self.nMaxBuf, (1,))        
        ## self.x_buf.append(-1.0)
        
        pass

        
    def update_buffer(self, X_test, Y_test):

        x          = X_test[-1]
        x_sup, idx = hdl.find_nearest(self.aXRange, x, sup=True)
        ## x_buf      = self.x_buf.get_array()

        mu_list  = [0.0]*self.nFutureStep
        var_list = [0.0]*self.nFutureStep

        # fXTOL should be sufficiently small.    
        if x - x_sup < self.fXTOL: # and x - x_buf[-1] >= 1.0:

            # obsrv_range X nFutureStep
            _, Y_pred_prob = self.ml.multi_step_approximated_predict(Y_test.tolist(),n_jobs=-1,full_step=True)

            for j in xrange(self.nFutureStep):
                (mu_list[j], var_list[j]) = hdl.gaussian_param_estimation(self.ml.obsrv_range, Y_pred_prob[:,j])
                self.buf_dict['mu_'+str(j)].append(mu_list[j])
                self.buf_dict['sig_'+str(j)].append(np.sqrt(var_list[j]))

            return mu_list, var_list, idx
        else:
            return None, None, idx

        
    def check_anomaly(self, y):

        var_coff = 1.0
        a_coff = np.zeros((self.nFutureStep))
        fAnomaly = self.fAnomaly
        
        for i in xrange(self.nFutureStep):

            # check buff size
            if len(self.buf_dict['mu_'+str(i)]) < i:
                a_coff[i] = 0.0
                fAnomaly -= 1.0
                continue
            
            mu  = self.buf_dict['mu_'+str(i)][0]
            sig = self.buf_dict['sig_'+str(i)][0]
            if mu-var_coff*sig > y or mu+var_coff*sig < y:
                a_coff[i] = 1.0
            else:
                a_coff[i] = 0.0

        score= sum(a_coff)
        ## print y, a_coff, score
        
        if score > fAnomaly:
            return True, score*(self.fAnomaly/fAnomaly)
        else:
            return False, score*(self.fAnomaly/fAnomaly)

        
    def simulation(self, X_test, Y_test, bReload):

        ## # Load data
        ## pkl_file = 'animation_data.pkl'
        ## if os.path.isfile(pkl_file) and bReload==False:
        ##     print "Load saved pickle"
        ##     data = ut.load_pickle(pkl_file)        
        ##     X_test      = data['X_test']
        ##     Y_test      = data['Y_test']
        ##     ## Y_pred      = data['Y_pred']
        ##     ## Y_pred_prob = data['Y_pred_prob']
        ##     mu          = data['mu']
        ##     var         = data['var']
        ## else:        

        ##     n = len(X_test)
        ##     mu = np.zeros((len(self.aXRange), self.nFutureStep))
        ##     var = np.zeros((len(self.aXRange), self.nFutureStep))

        ##     for i in range(1,n,1):
        ##         mu_list, var_list, idx = self.update_buffer(X_test[:i], Y_test[:i])

        ##         if mu_list != None and var_list != None:
        ##             mu[idx,:] = mu_list
        ##             var[idx,:]= var_list
                    
        ##     print "Save pickle"                    
        ##     data={}
        ##     data['X_test'] = X_test
        ##     data['Y_test'] = Y_test                
        ##     ## data['Y_pred'] = Y_pred
        ##     ## data['Y_pred_prob'] = Y_pred_prob
        ##     data['mu']          = mu
        ##     data['var']         = var
        ##     ut.save_pickle(data, pkl_file)                
        ## print "---------------------------"
            
        mu = np.zeros((len(self.aXRange), self.nFutureStep))
        var = np.zeros((len(self.aXRange), self.nFutureStep))

        self.fig = plt.figure(1)
        self.gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
                
        self.ax1 = self.fig.add_subplot(self.gs[0])
        self.ax1.set_xlim([0, X_test[-1].max()*1.2])
        self.ax1.set_ylim([0, max(self.ml.obsrv_range)*1.5])
        self.ax1.set_xlabel("Angle")
        self.ax1.set_ylabel("Force")

        lAll, = self.ax1.plot([], [], color='#66FFFF', lw=2)
        line, = self.ax1.plot([], [], lw=2)
        lmean, = self.ax1.plot([], [], 'm-', linewidth=2.0)    
        lvar1, = self.ax1.plot([], [], '--', color='0.75', linewidth=2.0)    
        lvar2, = self.ax1.plot([], [], '--', color='0.75', linewidth=2.0)    

        self.ax2 = self.fig.add_subplot(self.gs[1])        
        lbar,  = self.ax2.bar(0.0001, 0.0, width=1.0, color='b')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0, self.nMaxBuf])
        self.ax2.set_xlabel("Anomaly Gauge")        
        plt.setp(self.ax2.get_xticklabels(), visible=False)
        plt.setp(self.ax2.get_yticklabels(), visible=False)
        ## res_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        ## lbar2, = self.ax1.bar(30.0, 0.0, width=1.0, color='white', edgecolor='k')
        ## lvar , = self.ax1.fill_between([], [], [], facecolor='yellow', alpha=0.5)

        
        def init():
            lAll.set_data([],[])
            line.set_data([],[])
            lmean.set_data([],[])
            lvar1.set_data([],[])
            lvar2.set_data([],[])
            lbar.set_height(0.0)            
            return lAll, line, lmean, lvar1, lvar2, lbar,

        def animate(i):
            lAll.set_data(X_test, Y_test)            
            
            x = X_test[:i]
            y = Y_test[:i]
            line.set_data(x, y)

            if i > 0:
                mu_list, var_list, idx = self.update_buffer(x,y)            
                if mu_list != None and var_list != None:
                    mu[idx,:]  = mu_list
                    var[idx,:] = var_list

                ## # check anomaly score
                bFlag, fScore = self.check_anomaly(y[-1])
            
            if i >= 2 and i < len(Y_test):# -self.nFutureStep:

                x_sup = self.aXRange[idx]
                a_X  = np.arange(x_sup, x_sup+(self.nFutureStep+1)*self.fXInterval, self.fXInterval)
                
                if x[-1]-x_sup < x[-1]-x[-2]:                    
                    y_idx = 1
                else:
                    y_idx = int((x[-1]-x_sup)/(x[-1]-x[-2]))+1
                a_mu = np.hstack([y[-y_idx], mu[idx]])
                a_sig = np.hstack([0, np.sqrt(var[idx])])

                lmean.set_data( a_X, a_mu)
                lvar1.set_data( a_X, a_mu-1.*a_sig)
                lvar2.set_data( a_X, a_mu+1.*a_sig) 
                lbar.set_height(fScore)
                if fScore>=self.fAnomaly:
                    lbar.set_color('r')
                elif fScore>=self.fAnomaly*0.7:          
                    lbar.set_color('orange')
                else:
                    lbar.set_color('b')
                    
            else:
                lmean.set_data([],[])
                lvar1.set_data([],[])
                lvar2.set_data([],[])
                lbar.set_height(0.0)           
            return lAll, line, lmean, lvar1, lvar2, lbar,

           
        anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                                       frames=len(Y_test), interval=300, blit=True)

        anim.save('ani_test.mp4', fps=6, extra_args=['-vcodec', 'libx264'])
        
        plt.show()

        

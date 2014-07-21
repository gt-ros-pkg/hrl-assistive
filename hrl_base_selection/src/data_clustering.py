#!/usr/bin/env python

import roslib; roslib.load_manifest('hrl_base_selection')

# Common library
import numpy as np
import os.path
import sys

# ROS & Public library
#import rospy
import tf.transformations as tft
from sklearn.cluster import KMeans

# HRL library
import hrl_lib.util as ut
import hrl_lib.quaternion as qt

# Local library
import data_reader as dr
import kmeans as km

# Graphic library
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class DataCluster():

    def __init__(self, nCluster, minDist, nQuatCluster, minQuatDist):
        print 'Init DataCluster.'
        self.set_params(nCluster, minDist, nQuatCluster, minQuatDist)
        
    def set_params(self, nCluster, minDist, nQuatCluster, minQuatDist):
        self.nCluster = nCluster
        self.fMinDist = minDist

        self.nQuatCluster = nQuatCluster
        self.fMinQuatDist = minQuatDist
        
        self.ml = KMeans(n_clusters=nCluster, max_iter=300, n_jobs=6)
        
    def readData(self):
        print 'Read data manually.'
        data_start=0
        data_finish=1000 #'end'
        model = 'bed'
        subject='sub6_shaver'
        print 'Starting to convert data!'
        self.runData = dr.DataReader(subject=subject,data_start=data_start,data_finish=data_finish,model=model)

    def mat_to_pos_quat(self, raw_data):

        raw_pos  = np.zeros((len(raw_data),3)) #array
        raw_quat = np.zeros((len(raw_data),4))
        
        #-----------------------------------------------------------#
        ## Decompose data into pos,quat pairs
        for i in xrange(len(raw_data)):  
            raw_pos[i,:]  = np.array([raw_data[i][0,3],raw_data[i][1,3],raw_data[i][2,3]])
            raw_quat[i,:] = tft.quaternion_from_matrix(raw_data[i]) # order should be xyzw because ROS uses xyzw order.       
        
        return raw_pos, raw_quat

    def pos_clustering(self, raw_pos):

        while True:
            dict_params={}
            dict_params['n_clusters']=self.nCluster
            self.ml.set_params(**dict_params)
            self.ml.fit(raw_pos)

            # co-distance matrix
            bReFit = False
            co_pos_mat = np.zeros((self.nCluster,self.nCluster))
            for i in xrange(self.nCluster):

                # For refitting
                if bReFit == True: break
                
                for j in xrange(i, self.nCluster):
                    if i==j: 
                        co_pos_mat[i,j] = 1000000 # to avoid minimum check
                        continue
                    co_pos_mat[i,j] = co_pos_mat[j,i] = np.linalg.norm(self.ml.cluster_centers_[i] - self.ml.cluster_centers_[j])
                                        
                    if co_pos_mat[i,j] < self.fMinDist:
                        bReFit = True
                        break
                        
            if bReFit == True:
                self.nCluster -= 1
                print "New # of clusters: ", self.nCluster
                continue
            else:
                break
                    
        raw_pos_index = self.ml.fit_predict(raw_pos)
        return raw_pos_index

    # Return a list of clustered index.
    def grouping(self, raw_data):
        print 'Start clustering.'
        print raw_data.shape

        #-----------------------------------------------------------#
        ## Initialization
        raw_pos, raw_quat = self.mat_to_pos_quat(raw_data)
        
        #-----------------------------------------------------------#
        ## K-mean Clustring by Position
        raw_pos_index = self.pos_clustering(raw_pos)
        
        return raw_pos_index
        
    def clustering(self, raw_data):
        print 'Start clustering.'
        print raw_data.shape

        #-----------------------------------------------------------#
        ## Initialization
        raw_pos, raw_quat = self.mat_to_pos_quat(raw_data)

        #-----------------------------------------------------------#
        ## K-mean Clustering by Position
        raw_pos_index = self.pos_clustering(raw_pos)
        
        pos_clustered_group = []
        for i in xrange(self.nCluster):
            raw_group = []
            for j in xrange(len(raw_data)):
                if raw_pos_index[j] == i:
                    if raw_group == []:
                        raw_group = np.array([np.hstack([raw_pos[j],raw_quat[j]])])
                    else:
                        raw_group = np.vstack([raw_group, np.hstack([raw_pos[j],raw_quat[j]])])

            pos_clustered_group.append(raw_group)

        print "Number of pos groups: ", len(pos_clustered_group)
            
        #-----------------------------------------------------------#
        ## Grouping by orientation
        clustered_group = []        
        for group in pos_clustered_group:

            # samples
            X = group[:,3:]            
            ## print "Total X: ", X.shape[0], len(X)

            # Clustering parameters
            nQuatCluster = self.nQuatCluster
            kmsample = nQuatCluster  # 0: random centres, > 0: kmeanssample
            kmdelta = .001
            kmiter = 10
            metric = "quaternion"  # "chebyshev" = max, "cityblock" L1,  Lqmetric

            # the number of clusters should be smaller than the number of samples
            if nQuatCluster > len(X):
                nQuatCluster = len(X)
                kmsample = len(X)
                
            # Clustering
            while True:
                centres, xtoc, dist = km.kmeanssample( X, nQuatCluster, nsample=kmsample,
                                                    delta=kmdelta, maxiter=kmiter, metric=metric, verbose=0 )                          
        
                # co-distance matrix
                bReFit = False
                co_pos_mat = np.zeros((nQuatCluster,nQuatCluster))
                for i in xrange(nQuatCluster):

                    # For refitting
                    if bReFit == True: break
                    for j in xrange(i, nQuatCluster):
                        if i==j: 
                            co_pos_mat[i,j] = 1000000 # to avoid minimum check
                            continue
                        co_pos_mat[i,j] = co_pos_mat[j,i] = ut.quat_angle(centres[i],centres[j])                                         
                        if co_pos_mat[i,j] < self.fMinQuatDist:
                            bReFit = True
                            break

                if bReFit == True:
                    nQuatCluster -= 1
                    ## print "New # of clusters ", nQuatCluster, " in a sub group "
                    continue
                else:
                    break

            for i in xrange(nQuatCluster):
                raw_group = []
                for j in xrange(len(group)):
                    if xtoc[j] == i:
                        if raw_group == []:
                            raw_group = np.array([group[j,:]])
                        else:
                            raw_group = np.vstack([raw_group, group[j,:]])
                clustered_group.append(raw_group)

        print "Number of pos+quat groups: ", len(clustered_group)

        #-----------------------------------------------------------#
        ## Averaging
        avg_clustered_data = []
        num_clustered_data = []
        count = 0
        for i,g in enumerate(clustered_group):
            if len(g)==0: continue
            
            count += len(g)
            ## print "Number of sub samples: ", len(g)

            # Position
            pos_sum = np.array([0.,0.,0.])
            for j,s in enumerate(g):
                pos_sum += s[0]

                if j==0:
                    quat_array = s[1]
                else:
                    quat_array = np.vstack([quat_array, s[1]])
            pos_avg = pos_sum/float(len(g))

            # Quaternion
            quat_avg = qt.quat_avg( X )                                                                
            avg_clustered_data.append([pos_avg, quat_avg])
            num_clustered_data.append([len(g)])
                
        ## print "total: ", count
                  
        # Reshape the pairs into tranformation matrix
        for i, g in enumerate(avg_clustered_data):            

            mat = tft.quaternion_matrix(g[1])
            mat[0,3] = g[0][0]
            mat[1,3] = g[0][1]
            mat[2,3] = g[0][2]

            if i==0:
                clustered_data = np.array([mat])
            else:
                clustered_data = np.vstack([clustered_data,  np.array([mat])])    

        print "Final clustered data: ", clustered_data.shape, len(num_clustered_data)
        return clustered_data, num_clustered_data, len(pos_clustered_group)
             

    # X is a set of quaternion
    def q_image_axis_angle(self, X):

        print "Number of data: ", X.shape[0]
        
        angle_array = np.zeros((X.shape[0],1))
        direc_array = np.zeros((X.shape[0],3))
        
        for i in xrange(len(X)):
            angle, direc = qt.quat_to_angle_and_axis(X[i,:])
            angle_array[i,0] = angle
            direc_array[i,:] = direc

        # Normalize angles
        angle_array = (angle_array)/np.pi*180.0

        # matplot setup            
        fig = plt.figure(figsize=(12,12))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        ax = fig.add_subplot(111, projection='3d')
        
        # Plot a sphere
        r = 0.999
        u = np.linspace(0, 2 * np.pi, 120)
        v = np.linspace(0, np.pi, 60)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x*r, y*r, z*r,  rstride=1, cstride=1, color='c', alpha = 0.4, linewidth = 0)

        # Plot quaternions
        cmap = plt.cm.hsv        
        sc = ax.scatter(direc_array[:,0],direc_array[:,1],direc_array[:,2],c=angle_array,cmap=cmap,vmin=-180.0, vmax=180.0,s=100) #edgecolor='none'
        cbar = plt.colorbar(sc, ticks=np.arange(-180,180+30,30))
        
        ax.set_aspect("equal")
        ax.set_xlim([-1.0,1.0])
        ax.set_ylim([-1.0,1.0])
        ax.set_zlim([-1.0,1.0])
               
        font_dict={'fontsize': 30, 'family': 'serif'}        
        ax.set_xlabel('x', fontdict=font_dict)
        ax.set_ylabel('y', fontdict=font_dict)
        ax.set_zlabel('z', fontdict=font_dict)
        ax.view_init(20,80)
               
        plt.ion()    
        plt.show()
        #ax.mouse_init()
        ut.get_keystroke('Hit a key to proceed next')
                    
        return

    # X is a set of quaternion
    # Y is a set of label
    def q_image_axis_cluster(self, X, Y):

        print "Number of data: ", X.shape[0]
        
        angle_array = np.zeros((X.shape[0],1))
        direc_array = np.zeros((X.shape[0],3))
        
        for i in xrange(len(X)):
            angle, direc = qt.quat_to_angle_and_axis(X[i,:])
            angle_array[i,0] = angle
            direc_array[i,:] = direc

        ## # Normalize angles 
        angle_array = (angle_array)/np.pi*180.0
            
        # Normalize labels         
        max_label = float(np.max(Y))
        fY = np.zeros((len(Y),1))
        if max_label != 0:
            for i in xrange(len(Y)):
                fY[i] = float(Y[i])/max_label
            
        # matplot setup 
        fig = plt.figure(figsize=(24,12))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        #-------------- matplot 1 --------------
        ax = fig.add_subplot(121, projection='3d')
        font_dict={'fontsize': 45, 'family': 'serif'}            
        ax.set_title("QuTEM distribution", fontdict=font_dict)
        
        # Plot a sphere
        r = 1.0
        u = np.linspace(0, 2 * np.pi, 120)
        v = np.linspace(0, np.pi, 60)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x*r, y*r, z*r,  rstride=1, cstride=1, color='c', alpha = 0.4, linewidth = 0)

        # Plot quaternions
        cmap = plt.cm.hsv
        sc = ax.scatter(direc_array[:,0],direc_array[:,1],direc_array[:,2],c=angle_array,cmap=cmap,vmin=-180.0, vmax=180.0,s=100) #edgecolor='none'
        cbar = plt.colorbar(sc, ticks=np.arange(-180,180+30,30))
        ## cbar.set_clim(-180.0, 180.0)
        
        ax.set_aspect("equal")
        ax.set_xlim([-1.0,1.0])
        ax.set_ylim([-1.0,1.0])
        ax.set_zlim([-1.0,1.0])

        font_dict={'fontsize': 30, 'family': 'serif'}        
        ax.set_xlabel('x', fontdict=font_dict)
        ax.set_ylabel('y', fontdict=font_dict)
        ax.set_zlabel('z', fontdict=font_dict)
        ax.view_init(20,40)

        #-------------- matplot 2 --------------
        ax = fig.add_subplot(122, projection='3d')
        font_dict={'fontsize': 45, 'family': 'serif'}            
        ax.set_title("Clustering", fontdict=font_dict)
        
        # Plot a sphere
        r = 0.92
        u = np.linspace(0, 2 * np.pi, 120)
        v = np.linspace(0, np.pi, 60)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ## ax.plot_surface(x*r, y*r, z*r, rstride=1, cstride=1, color='c', alpha = 1.0, linewidth = 0)

        # Plot quaternions
        cmap = plt.cm.jet
        sc = ax.scatter(direc_array[:,0],direc_array[:,1],direc_array[:,2],c=Y,vmin=0,vmax=abs(Y).max(),s=100)
        ## plt.colorbar(sc)
        
        ax.set_aspect("equal")
        ax.set_xlim([-1.0,1.0])
        ax.set_ylim([-1.0,1.0])
        ax.set_zlim([-1.0,1.0])
        
        font_dict={'fontsize': 30, 'family': 'serif'}        
        ax.set_xlabel('x', fontdict=font_dict)
        ax.set_ylabel('y', fontdict=font_dict)
        ax.set_zlabel('z', fontdict=font_dict)
        ax.view_init(20,40)
               
        plt.ion()    
        plt.show()
        #ax.mouse_init()
        ut.get_keystroke('Hit a key to proceed next')
                    
        return
    

    def test(self, raw_data):
        print 'Start clustering.'
        print raw_data.shape

        N = 1000
        
        #-----------------------------------------------------------#
        ## Initialization
        raw_pos  = np.zeros((N,3)) #array
        raw_quat = np.zeros((N,4))
        
        #-----------------------------------------------------------#
        ## Decompose data into pos,quat pairs
        for i in xrange(N):            
            raw_pos[i,:]  = np.array([0,0,0])

        ## raw_quat = qt.quat_random( N )
        
        quat_mean = np.array([1.,0.,0.,1.5]);
        raw_quat = qt.quat_QuTem( quat_mean/np.linalg.norm(quat_mean), N, [0.03,0.3,0.3,1.0] )

        ## quat_mean = np.array([0.,1.,0.,-1.5]);
        ## raw_quat2 = qt.quat_QuTem( quat_mean/np.linalg.norm(quat_mean), N/2.0, [0.1,1.0,0.1,1.0] )
        ## raw_quat = np.vstack([raw_quat1,raw_quat2])

        ## raw_quat1 = np.array([[1.,  0.,  0.,  0.],
        ##                       [1.,  0.1, 0.,  0.],
        ##                       [1.,  0.,  0.1, 0.],
        ##                       [1.,  0.,  0.,  0.1],
        ##                       [1.,  0.2, 0.,  0.],
        ##                       [1.,  0.,  0.2, 0.],
        ##                       [1.,  0.,  0.,  0.2],
        ##                       [1.1, 0.1, 0.,  0.],
        ##                       [1.1, 0.,  0.1, 0.],
        ##                       [1.1, 0.,  0.,  0.1]])
        ## raw_quat2 = np.array([[0.,  0.,  1.,  0.],
        ##                       [0.1, 0.,  1.1, 0.],
        ##                       [0.1, 0.1, 1.,  0.],
        ##                       [0.,  1.,  0.,  0.],
        ##                       [0.1, 1.,  0.1, 0.],
        ##                       [0.1, 1.1, 0.,  0.],
        ##                       [0.1, 1.,  0.4, 0.],
        ##                       [0.1, 1.,  1.1, 0.2],
        ##                       [0.1, 1.,  1.4, 0.],
        ##                       [1.1, 1.,  0.1, 0.2],
        ##                       [1.1, 1.1,  0.1, 0.1]
        ##                       ])

        ## raw_quat = np.vstack([raw_quat1,raw_quat2])
        
        for i in xrange(len(raw_quat)):
            raw_quat[i,:] /= np.linalg.norm(raw_quat[i,:])
            
        #-----------------------------------------------------------#
        pos_clustered_group = []
        raw_group = np.hstack([raw_pos,raw_quat])
        pos_clustered_group.append(raw_group)

        print "Number of pos groups: ", len(pos_clustered_group)
            
        #-----------------------------------------------------------#
        ## Grouping by orientation
        clustered_group = []        
        for group in pos_clustered_group:

            # samples
            X = group[:,3:]            

            # Clustering parameters
            nQuatCluster = self.nQuatCluster
            kmsample = nQuatCluster  # 0: random centres, > 0: kmeanssample
            kmdelta = .001
            kmiter = 10
            metric = "quaternion"  # "chebyshev" = max, "cityblock" L1,  Lqmetric

            # the number of clusters should be smaller than the number of samples
            if nQuatCluster > len(X):
                nQuatCluster = len(X)
                kmsample = len(X)
                
            # Clustering
            while True:
                centres, xtoc, dist = km.kmeanssample( X, nQuatCluster, nsample=kmsample,
                                                       delta=kmdelta, maxiter=kmiter, metric=metric, verbose=0 )                                          
                # co-distance matrix
                bReFit = False
                co_pos_mat = np.zeros((nQuatCluster,nQuatCluster))
                for i in xrange(nQuatCluster):

                    # For refitting
                    if bReFit == True: break
                    for j in xrange(i, nQuatCluster):
                        if i==j: 
                            co_pos_mat[i,j] = 1000000 # to avoid minimum check
                            continue
                        co_pos_mat[i,j] = co_pos_mat[j,i] = ut.quat_angle(centres[i],centres[j])                                         
                        if co_pos_mat[i,j] < self.fMinQuatDist:
                            bReFit = True
                            break

                if bReFit == True:
                    nQuatCluster -= 1
                    ## print "New # of clusters ", nQuatCluster, " in a sub group "
                    continue
                else:
                    break

            ## for i in xrange(nQuatCluster):
            ##     raw_group = []
            ##     for j in xrange(len(group)):
            ##         if xtoc[j] == i:
            ##             if raw_group == []:
            ##                 raw_group = np.array([group[j,:]])
            ##             else:
            ##                 raw_group = np.vstack([raw_group, group[j,:]])
            ##     clustered_group.append(raw_group)

        print "Number of pos+quat groups: ", len(clustered_group)
        
        self.q_image_axis_cluster(X, xtoc)
        ## self.q_image_axis_angle(raw_quat) #temp

        #self.q_image_axis_angle(centres)
        
        
if __name__ == "__main__":

     dc = DataCluster(19,0.01,6,0.02)
     dc.readData()     
     ## dc.clustering(dc.runData.raw_goal_data)

     dc.set_params(2,0.01,6,0.02)
     index = dc.grouping(dc.runData.raw_goal_data)
     print index
     
     ## dc.test(dc.runData.raw_goal_data)

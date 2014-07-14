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

class DataCluster():

    def __init__(self, nCluster, minDist, nQuatCluster, minQuatDist):
        print 'Init DataCluster.'
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
    
    def clustering(self, raw_data):
        print 'Start clustering.'
        print raw_data.shape

        #-----------------------------------------------------------#
        ## Initialization
        raw_pos  = np.zeros((len(raw_data),3)) #array
        raw_quat = np.zeros((len(raw_data),4))

        #-----------------------------------------------------------#
        ## Decompose data into pos,quat pairs
        for i in xrange(len(raw_data)):            
            raw_pos[i,:]  = np.array([raw_data[i][0,3],raw_data[i][1,3],raw_data[i][2,3]])
            raw_quat[i,:] = tft.quaternion_from_matrix(raw_data[i]) # order? xyzw? wxyz?
          
        #-----------------------------------------------------------#
        ## K-mean Clustring by Position
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
        ## print raw_pos_index
        ## print self.ml.cluster_centers_

        pos_clustered_group = []
        for i in xrange(self.nCluster):
            raw_group = []
            for j in xrange(len(raw_data)):
                if raw_pos_index[j] == i:
                    raw_group.append([raw_pos[j],raw_quat[j]])
            
            pos_clustered_group.append(raw_group)

        print "Number of pos groups: ", len(pos_clustered_group)
            
        #-----------------------------------------------------------#
        ## Grouping by orientation
        clustered_group = []        
        for group in pos_clustered_group:

            # Taks samples
            X = np.array([group[0][1]])
            for i,s in enumerate(group):
                if i==0: continue
                X = np.vstack([X,s[1]])

            ## print "Total X: ", len(X)

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
                        raw_group.append(group[j])
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
        return clustered_data, num_clustered_data
             
            

if __name__ == "__main__":

     dc = DataCluster(19,0.01,5,0.02)
     dc.readData()
     dc.clustering(dc.runData.raw_goal_data)

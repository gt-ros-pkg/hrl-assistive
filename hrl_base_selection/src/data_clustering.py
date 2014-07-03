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

# Local library
import data_reader as dr

class DataCluster():

    def __init__(self, nCluster, minDist):
        print 'Init DataCluster.'
        self.nCluster = nCluster
        self.fMinDist = minDist
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

        ## Initialization
        raw_pos  = np.zeros((len(raw_data),3)) #array
        raw_quat = np.zeros((len(raw_data),4))

        ## Decompose data into pos,quat pairs
        for i in xrange(len(raw_data)):            
            raw_pos[i,:]  = np.array([raw_data[i][0,3],raw_data[i][1,3],raw_data[i][2,3]])
            raw_quat[i,:] = tft.quaternion_from_matrix(raw_data[i]) # order? xyzw? wxyz?
          
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
                    raw_group.append([raw_pos[i],raw_quat[i]])
            
            pos_clustered_group.append(raw_group)
                    
        # Grouping by orientation


        
        # Reshape the pairs into tranformation matrix

if __name__ == "__main__":

     dc = DataCluster(19,0.01)
     dc.readData()
     dc.clustering(dc.runData.raw_goal_data)

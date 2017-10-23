#!/usr/bin/env python
import sys
import os
import numpy as np
import cPickle as pkl
import random

# ROS
import roslib; roslib.load_manifest('hrl_pose_estimation')

# Graphics
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

# Machine Learning
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
## from skimage import data, color, exposure
from sklearn.decomposition import PCA

# HRL libraries
import hrl_lib.util as ut
import pickle
roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle

# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib
 


MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 

 
class DatabaseCreator():
    '''Gets the directory of pkl database and iteratively go through each file,
    cutting up the pressure maps and creating synthetic database'''
    def __init__(self, training_database_pkl_directory, save_pdf=False, verbose=False, select=False):

        # Set initial parameters
        self.training_dump_path = training_database_pkl_directory.rstrip('/')
        self.final_database_path = (
        os.path.abspath(os.path.join(self.training_dump_path, os.pardir)))

        self.verbose = verbose
        self.select = select
        self.world_to_mat = CreateDatasetLib().world_to_mat
        self.mat_to_taxels = CreateDatasetLib().mat_to_taxels

        if self.verbose: print 'The final database path is: ',self.final_database_path
        try:
            self.final_dataset = load_pickle(self.final_database_path+'/basic_train_dataset.p')
            print 'y'
        except IOError:
            print 'x'

        self.final_dataset = []


        print self.training_dump_path
        [self.p_world_mat, self.R_world_mat] = load_pickle('/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/mat_axes15.p')
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.individual_dataset = {}


        home_sup_dat = load_pickle(self.training_dump_path + '4/home_sup.p')
        if self.verbose: print "Checking database for empty values."
        empty_count = 0
        for entry in range(len(home_sup_dat)):
            home_joint_val = home_sup_dat[entry][1]
            if len(home_joint_val.flatten()) < (30) or (home_sup_dat[entry][0] < self.mat_size[0]*self.mat_size[1]):
                empty_count += 1
                del home_sup_dat[entry]

        if self.verbose: print "Empty value check results: {} rogue entries found".format(
            empty_count)

        # Targets in the mat frame
        home_sup_pressure_map = home_sup_dat[0][0]
        home_sup_joint_pos_world = home_sup_dat[0][1]
        home_sup_joint_pos = self.world_to_mat(home_sup_joint_pos_world, self.p_world_mat, self.R_world_mat)  # N x 3

        # print home_sup_joint_pos
        print len(home_sup_pressure_map)
        print len(home_sup_joint_pos), 'sizes'





    def visualize_single_pressure_map(self, p_map_raw, targets_raw=None):
        print p_map_raw.shape, 'pressure mat size'

        p_map = np.asarray(np.reshape(p_map_raw, self.mat_size))
        fig = plt.figure()

        # set options
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        xlim = [-10.0, 35.0]
        ylim = [70.0, -10.0]
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        # background
        ax1.set_axis_bgcolor('cyan')

        # Visualize pressure maps
        ax1.imshow(p_map, interpolation='nearest', cmap=
        plt.cm.bwr, origin='upper', vmin=0, vmax=100)

        # Visualize targets
        if targets_raw is not None:
            if type(targets_raw) == list:
                targets_raw = np.array(targets_raw)
            if len(np.shape(targets_raw)) == 1:
                targets_raw = np.reshape(targets_raw, (len(targets_raw) / 3, 3))

            target_coord = targets_raw[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax1.plot(target_coord[:, 0], target_coord[:, 1], 'y*', ms=8)
        plt.title('Pressure Mat and Targets')

        print targets_raw

        file_size = len(self.final_dataset) * 0.08958031837
        print 'output file size: ~', int(file_size), 'Mb'

        targets_raw_z = []
        for idx in targets_raw: targets_raw_z.append(idx[2])


        x = np.arange(0,10)
        ax2.bar(x, targets_raw_z)
        plt.xticks(x+0.5, ('Head', 'Torso', 'R Elbow', 'L Elbow', 'R Hand', 'L Hand', 'R Knee', 'L Knee', 'R Foot', 'L Foot'), rotation='vertical')
        plt.title('Distance above Bed')
        axkeep = plt.axes([0.01, 0.05, 0.08, 0.075])
        axdisc = plt.axes([0.01, 0.15, 0.08, 0.075])
        bdisc = Button(axdisc, 'Discard')
        bdisc.on_clicked(self.discard)
        bkeep = Button(axkeep, 'Keep')
        bkeep.on_clicked(self.keep)

        plt.show()

        return


    def discard(self, event):
        plt.close()
        self.keep_image = False

    def keep(self, event):
        plt.close()
        self.keep_image = True


    def create_raw_database(self):
        '''Creates a database using the raw pressure values(full_body) and only
        transforms world frame coordinates to mat coordinates'''

        for subject in [4]:
        #for subject in [4, 9, 15, 16, 17, 18]:
        #for subject in [4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:


            #self.training_dump_path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_'+str(subject)
            print self.training_dump_path

            home_sup = load_pickle(self.training_dump_path+str(subject)+'/home_sup.p')
            RH_sup = load_pickle(self.training_dump_path+str(subject)+'/RH_sup.p')
            LH_sup = load_pickle(self.training_dump_path+str(subject)+'/LH_sup.p')
            RL_sup = load_pickle(self.training_dump_path+str(subject)+'/RL_sup.p')
            LL_sup = load_pickle(self.training_dump_path+str(subject)+'/LL_sup.p')
        
            count = 0


            print 'working on subject ',subject, ' home_sup. ', 'length: ',len(home_sup)
            for [p_map_raw, target_raw] in home_sup:
                self.keep_image = False
                target_mat = self.world_to_mat(target_raw, self.p_world_mat, self.R_world_mat)
                rot_p_map = np.array(p_map_raw)
                rot_target_mat = target_mat
                self.visualize_single_pressure_map(rot_p_map, rot_target_mat)
                if self.keep_image == True: self.final_dataset.append([list(rot_p_map.flatten()), rot_target_mat.flatten()])
                count += 1

            print 'working on subject ',subject, ' LH sup. ', 'length: ',len(LH_sup)
        
            for [p_map_raw, target_raw] in LH_sup:
                print len(LH_sup) #100+, this is a list
                print len(LH_sup[4]) #2, this is a list
                print len(LH_sup[4][0]) #1728 #this is a list
                print LH_sup[4][1].shape  #this is an array
                break


                self.keep_image = False
                target_mat = self.world_to_mat(target_raw, self.p_world_mat, self.R_world_mat)
                rot_p_map = np.array(p_map_raw)
                rot_target_mat = target_mat
                if self.select == True:
                    self.visualize_single_pressure_map(rot_p_map, rot_target_mat)
                    if self.keep_image == True: self.final_dataset.append([list(rot_p_map.flatten()), rot_target_mat.flatten()])
                else:
                    self.final_dataset.append([list(rot_p_map.flatten()), rot_target_mat.flatten()])
           
                count += 1

            print 'working on subject ',subject, ' RH_sup. ', 'length: ',len(RH_sup)
            
            for [p_map_raw, target_raw] in RH_sup:
                self.keep_image = False
                target_mat = self.world_to_mat(target_raw, self.p_world_mat, self.R_world_mat)
                rot_p_map = np.array(p_map_raw)
                rot_target_mat = target_mat

                if self.select == True:
                    self.visualize_single_pressure_map(rot_p_map, rot_target_mat)
                    if self.keep_image == True: self.final_dataset.append([list(rot_p_map.flatten()), rot_target_mat.flatten()])
                else:
                    self.final_dataset.append([list(rot_p_map.flatten()), rot_target_mat.flatten()])
                count += 1

            print 'working on subject ',subject, ' LL_sup. ', 'length: ',len(LL_sup)
        
            for [p_map_raw, target_raw] in LL_sup:
                self.keep_image = False
                target_mat = self.world_to_mat(target_raw, self.p_world_mat, self.R_world_mat)
                rot_p_map = np.array(p_map_raw)
                rot_target_mat = target_mat
                if self.select == True:
                    self.visualize_single_pressure_map(rot_p_map, rot_target_mat)
                    if self.keep_image == True: self.final_dataset.append([list(rot_p_map.flatten()), rot_target_mat.flatten()])
                else:
                    self.final_dataset.append([list(rot_p_map.flatten()), rot_target_mat.flatten()])
                count += 1

            print 'working on subject ',subject, ' RL_sup. ', 'length: ',len(RL_sup)

            for [p_map_raw, target_raw] in RL_sup:
                self.keep_image = False
                target_mat = self.world_to_mat(target_raw, self.p_world_mat, self.R_world_mat)
                rot_p_map = np.array(p_map_raw)
                rot_target_mat = target_mat
                if self.select == True:
                    self.visualize_single_pressure_map(rot_p_map, rot_target_mat)
                    if self.keep_image == True: self.final_dataset.append([list(rot_p_map.flatten()), rot_target_mat.flatten()])
                else:
                    self.final_dataset.append([list(rot_p_map.flatten()), rot_target_mat.flatten()])
                count += 1



        print 'Output file size: ~', int(len(self.final_dataset) * 0.08958031837), 'Mb'
        print "Saving final_dataset"
        pkl.dump(self.final_dataset, open(os.path.join(self.training_dump_path+str(4),'../subject_x_dataset.p'), 'wb'))
            #pkl.dump(self.individual_dataset, open(os.path.join(self.training_dump_path, 'individual_database.p'), 'wb'))
        
        print 'Done.'
        return


    def run(self):
        '''Runs either the synthetic database creation script or the 
        raw dataset creation script to create a dataset'''
        self.create_raw_database()
        return





if __name__ == "__main__":

    import optparse
    p = optparse.OptionParser()

    p.add_option('--training_data_path', '--path',  action='store', type='string', \
                 dest='trainingPath',\
                 default='/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/subject_', \
                 help='Set path to the training database.')


    p.add_option('--lab_hd', action='store_true',
                 dest='lab_harddrive', \
                 default=False, \
                 help='Set path to the training database on lab harddrive.')

    p.add_option('--select', action='store_true', dest='select',
                 default=False, help='Presents visualization of all images for user to select discard/keep.')

    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=False, help='Printout everything (under construction).')
    
    opt, args = p.parse_args()

    print opt.trainingPath
    if opt.lab_harddrive == True:
        opt.trainingPath =  '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_'

    print opt.trainingPath
    
    #Initialize trainer with a training database file
    p = DatabaseCreator(training_database_pkl_directory=opt.trainingPath,verbose = opt.verbose, select = opt.select)
    p.run()
    sys.exit()

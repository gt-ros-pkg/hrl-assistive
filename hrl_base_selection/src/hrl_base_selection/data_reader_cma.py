#!/usr/bin/env python


import numpy as np
import math as m
import openravepy as op
import copy
import os.path

import time
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
import tf
from geometry_msgs.msg import PoseStamped
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from sensor_msgs.msg import JointState
from std_msgs.msg import String
import hrl_lib.transforms as tr
from hrl_base_selection.srv import BaseMove#, BaseMove_multi
from visualization_msgs.msg import Marker
from helper_functions import createBMatrix, Bmat_to_pos_quat
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from matplotlib.cbook import flatten

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle
#from sPickle import Pickler
import tf.transformations as tft
from score_generator_cma import ScoreGenerator
#import data_clustering as clust
import joblib
from joblib import Memory
from tempfile import mkdtemp
import hrl_lib.util as ut


class DataReader(object):
    
    def __init__(self, input_data=None, subject='sub6_shaver', reference_options=['head'],
                 data_start=0, data_finish=5, model='autobed', task='shaving',
                 pos_clust=5, ori_clust=1, tf_listener=None):
        self.score_sheet = []
        # if tf_listener is None:
        #     self.tf_listener = tf.TransformListener()
        # else:
        #     self.tf_listener = tf_listener
        if subject is None:
            self.subject = 0
            self.sub_num = 0
        else:
            self.subject = subject
            self.sub_num = int(list(subject)[3])
        self.data_start = data_start
        self.data_finish = data_finish

        self.model = model
        self.max_distance = 0.04 #10
        self.task = task
        self.num_goal_locations = 1
        self.pos_clust = pos_clust
        self.ori_clust = ori_clust

        self.input_data=input_data

        self.reference_options = reference_options

        self.clustered_goal_data = []
        self.clustered_number = []
        self.reference_options = []
        self.clustered_reference = []
        self.raw_goal_data = []
        self.goal_unique = []
        self.length = 0

    def receive_input_data(self, data, num, reference_options, reference):
        self.clustered_goal_data = data
        self.clustered_number = num
        self.reference_options = reference_options
        self.clustered_reference = reference
        self.length = len(num)
        self.raw_goal_data = data

    # Currently set to use only one reference frame, the head.
    def get_raw_data(self):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')
        #tool_init_pos = np.array([0.357766509056, 0.593838989735, 0.936517715454])
        #tool_init_rot = np.array([-0.23529052448, -0.502738550591, 3.00133908425])
        #head_init_pos = np.array([-0.142103180289, 0.457304298878, 1.13128328323]) 
        #head_init_rot = np.array([0.304673484999, -0.304466132626, 0.579344748594])
        #world_B_headinit = createBMatrix(head_init_pos,tft.quaternion_from_euler(head_init_rot[0],head_init_rot[1],head_init_rot[2],'rxyz'))
        #world_B_toolinit = createBMatrix(tool_init_pos,tft.quaternion_from_euler(tool_init_rot[0],tool_init_rot[1],tool_init_rot[2],'rxyz'))
        self.file_length = 0
        
        world_B_headc_reference_raw_data = np.array([map(float,line.strip().split()) for line in open(''.join([pkg_path,'/data/',self.subject,'_head.log']))])
        world_B_tool_raw_data = np.array([map(float,line.strip().split()) for line in open(''.join([pkg_path,'/data/',self.subject,'_tool.log']))]) 

        #dist = []
        #for num,line in enumerate(world_B_headc_reference_raw_data):
            #dist.append(np.linalg.norm(world_B_headc_reference_raw_data[num,1:4] - world_B_tool_raw_data[num,1:4]))
        #print 'raw distance: \n',np.min(np.array(dist))
        
        # max_length is the length of the shorter of the two files. They sometimes differ in length by a few points.
        max_length = np.min([len(world_B_headc_reference_raw_data),len(world_B_tool_raw_data)])

        # If data_finish is 'end' then it does from whatever is the start data to the end. self.length is the number of data points we are looking at.
        if self.data_finish == 'end':
            self.data_finish = max_length
        if self.data_finish > max_length:
            self.data_finish = max_length

        an = -m.pi/2
        tool_correction = np.matrix([[ m.cos(an),  0., m.sin(an),    0.], #.45 #.438
                                     [        0.,  1.,        0.,    0.], #0.34 #.42
                                     [-m.sin(an),  0., m.cos(an),    0.],
                                     [        0.,  0.,        0.,    1.]])


        self.length = np.min(np.array([max_length,self.data_finish-self.data_start]))
        

        world_B_headc_reference_data = np.zeros([self.length,4,4])
        world_B_tool_data = np.zeros([self.length,4,4])

        for num,line in enumerate(world_B_headc_reference_raw_data[self.data_start:self.data_finish]):
            world_B_headc_reference_data[num] = np.array(createBMatrix([line[1],line[2],line[3]],tft.quaternion_from_euler(line[4],line[5],line[6],'rxyz')))

        for num,line in enumerate(world_B_tool_raw_data[self.data_start:self.data_finish]):
            world_B_tool_data[num] = np.array(createBMatrix([line[1],line[2],line[3]],tft.quaternion_from_euler(line[4],line[5],line[6],'rxyz')))

        # I don't think data was collected on the tool tip, so I don't know what this data is. It has way more data than the others'
        #raw_tool_tip_data = np.array([map(float,line.strip().split()) for line in open(''.join([pkg_path,'/data/sub1_shaver_self_1_tool_tip.log']))])
        #tool_tip_data = np.zeros([len(raw_tool_tip_data),4,4])

        # We set the reference options here.
        self.reference_options = []
        self.reference_options.append('head') 
        self.reference = [] # Head references are associated with a value in self.reference of 0.



        #for num,line in enumerate(raw_tool_tip_data):
        #    tool_tip_data[num] = np.array(createBMatrix([line[2],line[3],line[4]],tft.quaternion_from_euler#(line[5],line[6],line[7],'rxyz')))
        self.distance = []
        self.raw_goal_data = [] #np.zeros([self.length,4,4])
        self.max_start_distance = .4 #0.2
        i = 0
        while self.raw_goal_data == []:
            #temp = np.array(np.matrix(world_B_headc_reference_data[i]).I*np.matrix(world_B_tool_data[i])*tool_correction)
            temp = np.array(np.matrix(world_B_headc_reference_data[i]).I*np.matrix(world_B_tool_data[i]))
            if np.linalg.norm(temp[0:3,3])<self.max_start_distance:
                self.raw_goal_data.append(temp)
                self.reference.append(0)
            else:
                i+=1
                print 'The first ', i, ' data points in the file was noise'

        for num in xrange(1,self.length):
            temp = np.array(np.matrix(world_B_headc_reference_data[num]).I*np.matrix(world_B_tool_data[num]))
            pos2, ori2 =Bmat_to_pos_quat(temp)
            pos1, ori1 =Bmat_to_pos_quat(self.raw_goal_data[len(self.raw_goal_data)-1])
           
            if np.linalg.norm(pos1-pos2)<self.max_distance:
                self.raw_goal_data.append(temp)
                self.reference.append(0)
                self.file_length+=1
                self.distance.append(np.linalg.norm(temp[0:3,3]))
        self.raw_goal_data = np.array(self.raw_goal_data)
        print 'Minimum distance between center of head and goal location from raw data = ', np.min(np.array(self.distance))
        return self.raw_goal_data

    def sample_raw_data(self, raw_data, num):
        sampled = []
        for i in xrange(num):
            sampled.append(raw_data[int(i*len(raw_data)/num)])
        return np.array(sampled)

    # Formats the goal data into a list where each entry is: [reference_B_goal, weight, reference]
    # Where reference_B_goal is the homogeneous transform from the reference coordinate frame to the goal frame
    # weight is generally 1 divided by the number of goals, used to give certain goals more weight
    # reference is the name of the coordinate frame that is the reference frame
    def generate_output_goals(self, test_goals=None, test_number=None, test_reference=None):
        if test_goals is None:
            goals = self.clustered_goal_data
            number = self.clustered_number
            reference = self.clustered_reference
        else:
            goals = test_goals
            if test_number is None:
                number = np.ones([len(goals), 1])
            else: number = test_number
            if test_reference is None:
                reference = np.zeros(len(goals))
            else:
                reference = test_reference
            print 'The number of raw goals being fed into the output goal generator is: ', len(test_goals)

        self.goal_unique = [] 
        for num in xrange(len(number)):
            self.goal_unique.append([goals[num], (float(number[num][0]))/(float(np.array(number).sum())),
                                     reference[num]])

        self.goal_unique = np.array(self.goal_unique)

        print 'Generated goals for score generator. There are ', len(self.goal_unique)
        
        return self.goal_unique

    def generate_score(self, viz_rviz=False, visualize=False, plot=False):
        cachedir = mkdtemp()
        # memory = Memory(cachedir=cachedir, verbose=0)
        memory = Memory(cachedir=cachedir, mmap_mode='r')
        self.num_base_locations = 1
        mytargets = 'all_goals'
        mytask = self.task
        myReferenceNames = self.reference_options
        myGoals = copy.copy(self.goal_unique)  # [self.data_start:self.data_finish]
        print 'There are ', len(myGoals), ' goals being sent to score generator.'
        selector = ScoreGenerator(visualize=visualize, targets=mytargets, reference_names=myReferenceNames,
                                  goals=myGoals, model=self.model, task=self.task)#, tf_listener=self.tf_listener)
        if viz_rviz:
            selector.show_rviz()
        score_sheet = selector.handle_score_generation(plot=plot)

        # default_is_zero = False
        # if not score_sheet[0., 0.]:
        #     print 'There are no good results in the score sheet'
        #     print score_sheet[0]
        # else:
        #     print 'The score sheet top item: '
        #     print '([heady, distance, x, y, theta, z, bed_z, bed_head_rest_theta, score])'
        #     print 'See the created pkl file for the entire set of data.'
        
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')

        if self.task == 'shaving' or True:
            # print 'Using the alternative streaming method for saving data because it is a big data set.'
            # filename = ''.join([pkg_path, '/data/', self.task, '_', self.model, '_subj_', str(self.sub_num),
            #                     '_score_data.pkl'])
            # filename = ''.join([pkg_path, '/data/', self.task, '_', self.model, '_cma_real_expanded_',
            filename = ''.join([pkg_path, '/data/', self.task, '_', self.model, '_cma_real',
                                # '_real_expanded_',
                                '_score_data.pkl'])
            save_pickle(score_sheet, filename)
            # filename = ''.join([pkg_path, '/data/', self.task, '_', self.model, '_subj_', str(self.sub_num),
            #                     '_score_data'])
            # joblib.dump(score_sheet, filename)
        else:
            save_pickle(score_sheet, ''.join([pkg_path, '/data/', self.task, '_', self.model, '_quick_score_data.pkl']))
        print 'I saved the data successfully!'
        return score_sheet

    def plot_score(self,load=False):
        # Plot the score as a scatterplot heat map
        #print 'score_sheet:',score_sheet
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')
        if load:
            print 'I am tring to load data to plot.'
            if os.path.isfile(''.join([pkg_path, '/data/',self.task,'_score_data.pkl'])):
                data=load_pickle(''.join([pkg_path, '/data/',self.task,'_score_data.pkl']))
                print 'Successful load. Will plot the latest and greatest version of the score sheet.'
            else:
                print 'I am only plotting the data that I just processed. I could not find what to load.'
                data=self.score_sheet
        else:
            print 'I am only plotting the data that I just processed. I did not load anything to plot.'
            data=self.score_sheet
        score2d_temp = []
        #print t
        for i in np.arange(-1.5,1.55,.05):
            for j in np.arange(-1.5,1.55,.05):
                temp = []
                for item in data:
                    newline = []
                #print 'i is:',i
                #print 'j is:',j
                    if item[0]==i and item[1]==j:
                        newline.append([i,j,item[3]])
                        for k in xrange(self.num_base_locations):
                            newline.append(item[int(4 + 2*k)])
                            newline.append(item[int(5 + 2*k)])
                        #print 'newest line ',list(flatten(newline))
                        temp.append(list(flatten(newline)))
                temp=np.array(temp)
                temp_max = []
                temp_max.append(np.max(temp[:,2]))
                for k in xrange(self.num_base_locations):
                    temp_max.append(np.max(temp[:,int(3+2*k)]))
                    temp_max.append(np.max(temp[:,int(4+2*k)]))
                #print 'temp_max is ',temp_max
                score2d_temp.append(list(flatten([i,j,temp_max])))

        #print '2d score:',np.array(score2d_temp)[0]

        seen_items = []
        score2d = [] 
        for item in score2d_temp:
            #any((a == x).all() for x in my_list)
            #print 'seen_items is: ',seen_items
            #print 'item is: ',item
            #print (any((item == x) for x in seen_items))
            if not (any((item == x) for x in seen_items)):
            #if item not in seen_items:
                #print 'Just added the item to score2d'
                score2d.append(item)
                seen_items.append(item)
        score2d = np.array(score2d)
        #print 'score2d with no repetitions',score2d
        #for item in score2d:
        #    if (item[0]<0.5) and (item[0]>-0.5) and (item[1]<0.5) and (item[1]>-0.5):
        #        print item


        if self.model == 'chair':
            verts_subject = [(-.438, -.32885), # left, bottom
                             (-.438, .32885), # left, top
                             (.6397, .32885), # right, top
                             (.6397, -.32885), # right, bottom
                             (0., 0.), # ignored
                        ]
        elif self.model == 'bed':
            verts_subject = [(-.2954, -.475), # left, bottom
                             (-.2954, .475), # left, top
                             (1.805, .475), # right, top
                             (1.805, -.475), # right, bottom
                             (0., 0.), # ignored
                            ]
        elif self.model == 'autobed':
            verts_subject = [(-.2954, -.475), # left, bottom
                             (-.2954, .475), # left, top
                             (1.805, .475), # right, top
                             (1.805, -.475), # right, bottom
                             (0., 0.), # ignored
                            ]
        
        
        verts_pr2 = [(-1.5,  -1.5), # left, bottom
                     ( -1.5, -.835), # left, top
                     (-.835, -.835), # right, top
                     (-.835,  -1.5), # right, bottom
                     (   0.,    0.), # ignored
                    ]
    
        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY,
                ]
           
        path_subject = Path(verts_subject, codes)
        path_pr2 = Path(verts_pr2, codes)
    
        patch_subject = patches.PathPatch(path_subject, facecolor='orange', lw=2)        
        patch_pr2 = patches.PathPatch(path_pr2, facecolor='orange', lw=2)
        
        X  = score2d[:,0]
        Y  = score2d[:,1]
        #Th = score_sheet[:,2]
        c3  = score2d[:,2]

        fig3 = plt.figure(1)
        ax3 = fig3.add_subplot(111)
        surf3 = ax3.scatter(X, Y,s=60, c=c3,alpha=1)
        ax3.set_xlabel('X Axis')
        ax3.set_ylabel('Y Axis')
        fig3.colorbar(surf3, shrink=0.65, aspect=5)
        ax3.add_patch(patch_subject)
        ax3.add_patch(patch_pr2)
        ax3.set_xlim(-2,2)
        ax3.set_ylim(-2,2)
        fig3.set_size_inches(14,11,forward=True)
        if load:
            ax3.set_title(''.join(['Plot of personal space score from ',self.task]))
            plt.savefig(''.join([pkg_path, '/images/space_score_of_',self.task,'.png']), bbox_inches='tight')     
        else:
            ax3.set_title(''.join(['Plot of personal space score from ',self.subject,' on a ',self.model,' model.',' Data: (',str(self.data_start),' - ',str(self.data_finish),')']))
            plt.savefig(''.join([pkg_path, '/images/space_score_of_',self.model,'_',self.subject,'_numbers_',str(self.data_start),'_',str(self.data_finish),'.png']), bbox_inches='tight')


        print 'Number of base locations is: ',self.num_base_locations
        #print 'score2d ',score2d[0,:]
        for i in xrange(self.num_base_locations):
            
            c = copy.copy(score2d[:,3+2*i])
            c2 = copy.copy(score2d[:,4+2*i])

            fig = plt.figure(2+2*i)
            ax = fig.add_subplot(111)
            surf = ax.scatter(X, Y, s=60,c=c,alpha=1)
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            fig.colorbar(surf, shrink=0.65, aspect=5)
            ax.add_patch(patch_subject)
            ax.add_patch(patch_pr2)
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
            fig.set_size_inches(14,11,forward=True)
            if load:
                ax.set_title(''.join(['Plot of reach score from BP',str(i+1),' ',self.task]))
                plt.savefig(''.join([pkg_path, '/images/reach_score_of_BP',str(i+1),'_',self.task,'.png']), bbox_inches='tight')       
            else:
                ax.set_title(''.join(['Plot of reach score from BP',str(i+1),' ',self.subject,' on a ',self.model,' model.',' Data: (',str(self.data_start),' - ',str(self.data_finish),')']))
                plt.savefig(''.join([pkg_path, '/images/reach_score_of_BP',str(i+1),'_',self.model,'_',self.subject,'_numbers_',str(self.data_start),'_',str(self.data_finish),'.png']), bbox_inches='tight')
            
            fig2 = plt.figure(3+2*i)
            ax2 = fig2.add_subplot(111)
            surf2 = ax2.scatter(X, Y,s=60, c=c2,alpha=1)
            ax2.set_xlabel('X Axis')
            ax2.set_ylabel('Y Axis')
            fig2.colorbar(surf2, shrink=0.65, aspect=5)
            ax2.add_patch(patch_subject)
            ax2.add_patch(patch_pr2)
            ax2.set_xlim(-2,2)
            ax2.set_ylim(-2,2)
            fig2.set_size_inches(14,11,forward=True)
            if load:
                ax2.set_title(''.join(['Plot of manipulability score from BP',str(i+1),' ',self.task]))
                plt.savefig(''.join([pkg_path, '/images/manip_score_of_BP',str(i+1),'_',self.task,'.png']), bbox_inches='tight')     
            else:
                ax2.set_title(''.join(['Plot of manip score from BP',str(i+1),' ',self.subject,' on a ',self.model,' model.',' Data: (',str(self.data_start),' - ',str(self.data_finish),')']))
                plt.savefig(''.join([pkg_path, '/images/manip_score_of_BP',str(i+1),'_',self.model,'_',self.subject,'_numbers_',str(self.data_start),'_',str(self.data_finish),'.png']), bbox_inches='tight')

        #plt.ion()
        plt.show()                                                                                                
        #ut.get_keystroke('Hit a key to proceed next')

    def plot_goals(self):
        fig = plt.figure()
        #fig = plt.gcf()
        fig.set_size_inches(14,11,forward=True)
        ax = fig.add_subplot(111, projection='3d')
        X  = self.clustered_goal_data[:,0,3]
        Y  = self.clustered_goal_data[:,1,3]
        Z = self.clustered_goal_data[:,2,3]
        #print X,len(X),Y,len(Y),Z,len(Z)
        c  = 'b'
        surf = ax.scatter(X, Y, Z,s=40, c=c,alpha=.5)
        #surf = ax.scatter(X, Y,s=40, c=c,alpha=.6)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        ax.set_title(''.join(['Plot of goals from ',self.subject,'. Data: (',str(self.data_start),' - ',str(self.data_finish),')']))

        #fig.colorbar(surf, shrink=0.5, aspect=5)
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')
        ax.set_xlim(-.2,.2)
        ax.set_ylim(-.2,.2)
        ax.set_zlim(-.2,.2)
        #plt.savefig(''.join([pkg_path, '/images/goals_plot_',self.model,'_',self.subject,'_numbers_',str(self.data_start),'_',str(self.data_finish),'.png']), bbox_inches='tight')
        
        plt.ion()                                                                                                 
        plt.show()                                                                                                
        ut.get_keystroke('Hit a key to proceed next')

    def pub_rviz(self):
        self.num_base_locations = 1
        mytargets = 'all_goals'
        mytask = self.task #'shaving'
        #start = 0
        #end = 3
        myGoals = copy.copy(self.goal_unique)#[self.data_start:self.data_finish]
        print 'There are ',len(myGoals),' goals being sent to score generator.'
        #print myGoals
        selector = ScoreGenerator(visualize=False,targets=mytargets,goals = myGoals,model=self.model)#,tf_listener=self.tf_listener)
        selector.show_rviz()


if __name__ == "__main__":
    data_start = 0
    data_finish = 'end' #'end'
    model = 'chair' #options are: 'chair', 'bed', 'autobed'
    task = 'shaving'
    subject = 'sub6_shaver'
    pos_clust = 50
    ori_clust = 2
    rospy.init_node(''.join(['data_reader_', subject, '_', str(data_start), '_', str(data_finish), '_',
                             str(int(time.time()))]))
    start_time = time.time()
    print 'Starting to convert data!'
    runData = DataReader(subject=subject, data_start=data_start, data_finish=data_finish, model=model, task=task,
                         pos_clust=pos_clust, ori_clust=ori_clust)
    raw_data = runData.get_raw_data()

    ## To test clustering by using raw data sampled instead of clusters
    # sampled_raw = runData.sample_raw_data(raw_data,1000)
    # runData.generate_output_goals(test_goals=sampled_raw)
    
    # To run using the clustering system
    runData.cluster_data()
    runData.generate_output_goals()

    print 'Time to convert data into useful matrices: %fs'%(time.time()-start_time)
    print 'Now starting to generate the score. This will take a long time if there were many goal locations.'
    start_time = time.time()
    runData.pub_rviz()
    #runData.plot_goals()
    # runData.generate_score(viz_rviz=True, visualize=False, plot=False)
    print 'Time to generate all scores: %fs' % (time.time()-start_time)
    #print 'Now trying to plot the data. This might take a while for lots of data; depends on amount of data in score sheet. ~60 seconds.'
    #start_time = time.time()
    #runData.plot_score(load=False)
    #print 'Time to plot score: %fs'%(time.time()-start_time)
    rospy.spin()


























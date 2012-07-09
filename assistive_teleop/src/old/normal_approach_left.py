#!/usr/bin/python

import roslib; roslib.load_manifest('web_teleop_trunk')
import rospy
import math
from geometry_msgs.msg  import PoseStamped
from std_msgs.msg import Float32, String
from tf import TransformListener, transformations, TransformBroadcaster

class NormalApproachLeft():
    
    standoff = 0.368 #0.2 + 0.168 (dist from wrist to fingertips)
    frame = 'base_footprint'
    px = py = pz = 0;
    qx = qy = qz = 0;
    qw = 1;

    def __init__(self):
        rospy.init_node('normal_approach_left')
        rospy.Subscriber('norm_approach_left', PoseStamped, self.update_frame)
        rospy.Subscriber('l_hand_pose', PoseStamped, self.update_curr_pose)
        rospy.Subscriber('wt_lin_move_left',Float32, self.linear_move)
        self.goal_out = rospy.Publisher('wt_left_arm_pose_commands', PoseStamped)
        self.move_arm_out = rospy.Publisher('wt_move_left_arm_goals', PoseStamped)
        self.test_out = rospy.Publisher('left_hand_test_pose', PoseStamped)
        self.tf = TransformListener()
        self.tfb = TransformBroadcaster()
        self.wt_log_out = rospy.Publisher('wt_log_out', String )

    def update_curr_pose(self, msg):
        self.currpose = msg;

    def update_frame(self, pose):

        self.standoff = 0.368
        self.frame = pose.header.frame_id
        self.px = pose.pose.position.x    
        self.py = pose.pose.position.y    
        self.pz = pose.pose.position.z    
        self.qx = pose.pose.orientation.x
        self.qy = pose.pose.orientation.y
        self.qz = pose.pose.orientation.z
        self.qw = pose.pose.orientation.w

        self.tfb.sendTransform((self.px,self.py,self.pz),(self.qx,self.qy,self.qz,self.qw), rospy.Time.now(), "left_pixel_3d_frame", self.frame)
        self.find_approach(pose)

    def find_approach(self, msg):
            self.pose_in = msg
            self.tf.waitForTransform('left_pixel_3d_frame','base_footprint', rospy.Time(0), rospy.Duration(3.0))
            self.tfb.sendTransform((self.pose_in.pose.position.x, self.pose_in.pose.position.y, self.pose_in.pose.position.z),(self.pose_in.pose.orientation.x, self.pose_in.pose.orientation.y, self.pose_in.pose.orientation.z, self.pose_in.pose.orientation.w), rospy.Time.now(), "left_pixel_3d_frame", self.pose_in.header.frame_id)

            self.tf.waitForTransform('left_pixel_3d_frame','l_wrist_roll_link', rospy.Time.now(), rospy.Duration(3.0))
            
            goal = PoseStamped()
            goal.header.frame_id = 'left_pixel_3d_frame'
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.z = self.standoff
            goal.pose.orientation.x = 0#.5*math.sqrt(2)
            goal.pose.orientation.y = 0.5*math.sqrt(2)
            goal.pose.orientation.z = 0#.5*math.sqrt(2)
            goal.pose.orientation.w = 0.5*math.sqrt(2)
            #print "Goal:\r\n %s" %goal    

            self.tf.waitForTransform(goal.header.frame_id, 'torso_lift_link', rospy.Time.now(), rospy.Duration(3.0))
            appr = self.tf.transformPose('torso_lift_link', goal)
            #print "Appr: \r\n %s" %appr    

            self.wt_log_out.publish(data="Normal Approach with left hand: Trying to move WITH motion planning")
            self.move_arm_out.publish(appr)
            #self.test_out.publish(appr)

    def linear_move(self, msg):
        print "Linear Move: Left Arm: %s m Step" %msg.data
        self.tf.waitForTransform(self.currpose.header.frame_id, 'l_wrist_roll_link', self.currpose.header.stamp, rospy.Duration(3.0))
        newpose = self.tf.transformPose('l_wrist_roll_link', self.currpose)
        newpose.pose.position.x += msg.data
        step_goal = self.tf.transformPose(self.currpose.header.frame_id, newpose)
        self.goal_out.publish(step_goal)
        
if __name__ == '__main__':
    NAL = NormalApproachLeft()

    r = rospy.Rate(10)    
    while not rospy.is_shutdown():
        NAL.tfb.sendTransform((NAL.px,NAL.py,NAL.pz),(NAL.qx,NAL.qy,NAL.qz,NAL.qw), rospy.Time.now(), "left_pixel_3d_frame", NAL.frame)
        r.sleep()

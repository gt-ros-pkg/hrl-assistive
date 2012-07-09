#!/usr/bin/python

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
import actionlib
from geometry_msgs.msg  import PoseStamped
from std_msgs.msg import String, Bool
from trajectory_msgs.msg import JointTrajectoryPoint
from pr2_controllers_msgs.msg import JointTrajectoryControllerState, JointTrajectoryActionGoal
from pr2_common_action_msgs.msg import TuckArmsAction, TuckArmsGoal


class TuckArmsIntermediary():

    def __init__(self):
            rospy.init_node('tuck_arms_intermediary')
            self.tuck = actionlib.SimpleActionClient('tuck_arms', TuckArmsAction)
            rospy.Subscriber('r_arm_controller/state', JointTrajectoryControllerState , self.set_joint_state_r)
            rospy.Subscriber('l_arm_controller/state', JointTrajectoryControllerState , self.set_joint_state_l)
            rospy.Subscriber('wt_tuck_arms', String, self.tuck_arm)
            self.joints_out_r = rospy.Publisher('r_arm_controller/joint_trajectory_action/goal', JointTrajectoryActionGoal )
            self.joints_out_l = rospy.Publisher('l_arm_controller/joint_trajectory_action/goal', JointTrajectoryActionGoal )
            self.wt_log_out = rospy.Publisher('wt_log_out', String )

            print "Waiting for tuck_arms server"
            self.tuck.wait_for_server()    
            print "Tuck arms server found"    
    
    def set_joint_state_r(self,msg):
            self.joint_state_r = msg

    def set_joint_state_l(self,msg):
            self.joint_state_l = msg
    
    def tuck_arm(self, msg):
        tuck_goal = TuckArmsGoal()
        
        if (msg.data == 'right' or msg.data == 'left'):
            recover = True
            recover_goal = JointTrajectoryActionGoal()
        else:
            recover = False

        if msg.data == 'right':
            tuck_goal.tuck_left = False
            tuck_goal.tuck_right = True
            print "Tucking Right Arm Only"
            self.wt_log_out.publish(data="Tucking Right Arm Only")
           
            recover_goal.goal.trajectory.joint_names = self.joint_state_l.joint_names
            recover_position = self.joint_state_l.actual
            recover_position.time_from_start = rospy.Duration(5) 
            recover_publisher = self.joints_out_l

        elif msg.data == 'left':
            tuck_goal.tuck_left = True
            tuck_goal.tuck_right = False
            print "Tucking Left Arm Only"
            self.wt_log_out.publish(data="Tucking Left Arm Only")

            recover_goal.goal.trajectory.joint_names = self.joint_state_r.joint_names
            recover_position = self.joint_state_r.actual
            recover_position.time_from_start = rospy.Duration(5) 
            recover_publisher = self.joints_out_r 
        elif msg.data == 'both':
            tuck_goal.tuck_left = True
            tuck_goal.tuck_right = True
            print "Tucking Both Arms"
            self.wt_log_out.publish(data="Tucking Both Arms")

        else:
           print "Bad input to Tuck Arm Intermediary"
        

        finished_within_time = False
        self.tuck.send_goal(tuck_goal)
        finished_within_time = self.tuck.wait_for_result(rospy.Duration(30))
        if not (finished_within_time):
            self.tuck.cancel_goal()
            self.wt_log_out.publish(data="Timed out tucking right arm")
            rospy.loginfo("Timed out tucking right arm")
        else:
            state = self.tuck.get_state()
            success = (state == 'SUCCEEDED')
            if (success):
                rospy.loginfo("Action Finished: %s" %state)
                self.wt_log_out.publish(data="Tuck Right Arm: Succeeded")
            else:
                rospy.loginfo("Action failed: %s" %state)
                self.wt_log_out.publish(data="Tuck Right Arm: Failed: %s" %state)

        if (recover):
            recover_goal.goal.trajectory.points.append(recover_position)
            recover_publisher.publish(recover_goal);
            print "Sending Recovery Goal to restore initial position of non-tucked arm"
            self.wt_log_out.publish(data="Sending Recovery Goal to restore initial position of non-tucked arm")
        
if __name__ == '__main__':
    TAI = TuckArmsIntermediary()

    while not rospy.is_shutdown():
        rospy.sleep(0.1)

#! /usr/bin/env python

PKG = 'web_teleop_trunk'

import roslib; roslib.load_manifest(PKG)
import rospy
import pickle

from geometry_msgs.msg import PoseStamped
from kinematics_msgs.srv import GetConstraintAwarePositionIK, GetConstraintAwarePositionIKRequest, GetKinematicSolverInfo, GetKinematicSolverInfoRequest, GetPositionIK, GetPositionIKRequest
from arm_navigation_msgs.srv import GetStateValidity,GetStateValidityRequest, SetPlanningSceneDiff, SetPlanningSceneDiffRequest

x_steps = 2
y_steps = 2
z_steps = 2

class ik_test():

    def __init__(self):
        rospy.init_node('ik_test')

        self.ik_info = '/pr2_right_arm_kinematics/get_ik_solver_info'
        self.set_planning_scene_diff_name = '/environment_server/set_planning_scene_diff'
        
        print "waiting for ik service"
        rospy.wait_for_service(self.ik_info)
        self.ik_info_service = rospy.ServiceProxy(self.ik_info, GetKinematicSolverInfo)
        ik_info_req = GetKinematicSolverInfoRequest()
        self.ik_info = self.ik_info_service.call(ik_info_req)
        #print "IK INFO:"
        #print self.ik_info
        
        #print "waiting for planning scene service"
        #rospy.wait_for_service(self.set_planning_scene_diff_name)
        #self.set_planning_scene_diff = rospy.ServiceProxy(self.set_planning_scene_diff_name, SetPlanningSceneDiff)
        #self.set_planning_scene_diff.call(SetPlanningSceneDiffRequest())

        self.ik_service = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_ik', GetPositionIK, True)

        self.ik_req = GetPositionIKRequest()
        self.ik_req.timeout = rospy.Duration(1)
        self.ik_req.ik_request.ik_link_name = self.ik_info.kinematic_solver_info.link_names[-1]
        self.ik_req.ik_request.ik_seed_state.joint_state.name = self.ik_info.kinematic_solver_info.joint_names
        self.ik_req.ik_request.ik_seed_state.joint_state.position =  [0]*7
        
        self.ps = PoseStamped()
        self.ps.header.stamp = rospy.Time(0)
        self.ps.header.frame_id = 'torso_lift_link'
        self.ps.pose.position.x = self.ps.pose.position.y = self.ps.pose.position.z = 0
        self.ps.pose.orientation.x = self.ps.pose.orientation.y = self.ps.pose.orientation.z = 0
        self.ps.pose.orientation.w = 1

        self.results = [[88 for i in xrange(x_steps*y_steps*z_steps)] , [PoseStamped() for i in xrange(x_steps*y_steps*z_steps)]] 
        print len(self.results[0])
        self.count = 0
        self.run_poses()


    def run_poses(self):
        for i in xrange(x_steps):
            print i, "/50\n"
            self.ps.pose.position.x = float(i)/50
            for j in xrange(y_steps):
                #print j, "/ 100\n"
                self.ps.pose.position.y = -1 + float(j)/50
                for k in xrange(z_steps):
                    self.ps.pose.position.z = -1 + float(k)/50
                    print self.count
                    self.get_ik(self.ps, self.count)
                 

    def get_ik(self, ps, count):
        self.ik_req.ik_request.pose_stamped = ps
        ik_goal = self.ik_service(self.ik_req)
        self.results[0][count] = ik_goal.error_code.val
        self.results[1][count] = ps
        self.count += 1



#class TestPr2ArmKinematicsWithCollisionObjects(unittest.TestCase):
#
#    def generateRandomValues(self):
#        ret = [float() for _ in range(len(self.min_limits))]
#        for i in range(len(self.min_limits)):
#            ret[i] = random.uniform(self.min_limits[i], self.max_limits[i])
#        return ret
#
#    def setUp(self):
#
#        random.seed()
#
#        self.coll_ik_name = '/pr2_right_arm_kinematics/get_constraint_aware_ik'
#        self.env_server_name = '/planning_scene_validity_server/get_state_validity'
#        self.ik_info = '/pr2_right_arm_kinematics/get_ik_solver_info'
#        self.set_planning_scene_diff_name = '/environment_server/set_planning_scene_diff'
#
#        rospy.init_node('test_pr2_arm_kinematics_with_constraints')
#
#        self.att_pub = rospy.Publisher('attached_collision_object',AttachedCollisionObject,latch=True)
#        self.obj_pub = rospy.Publisher('collision_object',CollisionObject,latch=True)
#
#        rospy.wait_for_service(self.coll_ik_name)
#        self.ik_service = rospy.ServiceProxy(self.coll_ik_name, GetConstraintAwarePositionIK)
#
#        rospy.wait_for_service(self.ik_info)
#        self.ik_info_service = rospy.ServiceProxy(self.ik_info, GetKinematicSolverInfo)
#        
#        rospy.wait_for_service(self.set_planning_scene_diff_name)
#        self.set_planning_scene_diff = rospy.ServiceProxy(self.set_planning_scene_diff_name, SetPlanningSceneDiff)
#
#        rospy.wait_for_service(self.env_server_name)
#        self.state_validity = rospy.ServiceProxy(self.env_server_name, GetStateValidity)
#
#        self.joint_names = ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint']
#
#        self.state_req = GetStateValidityRequest()
#        self.state_req.group_name = 'right_arm'
#        self.state_req.robot_state.joint_state.name = self.joint_names
#        self.state_req.robot_state.joint_state.position = [float(0.0) for _ in range(7)]
#        self.state_req.check_collisions = True
#
#        ik_info_req = GetKinematicSolverInfoRequest()
#        ik_info_res = self.ik_info_service.call(ik_info_req)
#        self.min_limits = [float() for _ in range(len(ik_info_res.kinematic_solver_info.limits))]
#        self.max_limits = [float() for _ in range(len(ik_info_res.kinematic_solver_info.limits))]
#        for i in range(len(ik_info_res.kinematic_solver_info.limits)):
#            self.min_limits[i] = ik_info_res.kinematic_solver_info.limits[i].min_position
#            self.max_limits[i] = ik_info_res.kinematic_solver_info.limits[i].max_position
#        
#        print len(self.min_limits), len(self.max_limits)
#
#        self.table = CollisionObject()
#         
#        self.table.header.stamp = rospy.Time.now()
#        self.table.header.frame_id = "base_link"
#        self.table.id = "table";
#        self.table.operation.operation = arm_navigation_msgs.msg.CollisionObjectOperation.ADD
#        self.table.shapes = [Shape() for _ in range(1)]
#        self.table.shapes[0].type = Shape.BOX
#        self.table.shapes[0].dimensions = [float() for _ in range(3)]
#        self.table.shapes[0].dimensions[0] = 1.0
#        self.table.shapes[0].dimensions[1] = 1.0
#        self.table.shapes[0].dimensions[2] = .05
#        self.table.poses = [Pose() for _ in range(1)]
#        self.table.poses[0].position.x = 1.0
#        self.table.poses[0].position.y = 0
#        self.table.poses[0].position.z = .5
#        self.table.poses[0].orientation.x = 0
#        self.table.poses[0].orientation.y = 0
#        self.table.poses[0].orientation.z = 0
#        self.table.poses[0].orientation.w = 1 
#
#        self.att_box = AttachedCollisionObject()
#        self.att_box.link_name = 'r_gripper_palm_link'
#        self.att_box.object.header.stamp = rospy.Time.now()
#        self.att_box.object.header.frame_id = "r_gripper_palm_link"
#        self.att_box.object.id = "att_box.object";
#        self.att_box.object.operation.operation = arm_navigation_msgs.msg.CollisionObjectOperation.ADD
#        self.att_box.object.shapes = [Shape() for _ in range(1)]
#        self.att_box.object.shapes[0].type = Shape.BOX
#        self.att_box.object.shapes[0].dimensions = [float() for _ in range(3)]
#        self.att_box.object.shapes[0].dimensions[0] = .04
#        self.att_box.object.shapes[0].dimensions[1] = .04
#        self.att_box.object.shapes[0].dimensions[2] = .2
#        self.att_box.object.poses = [Pose() for _ in range(1)]
#        self.att_box.object.poses[0].position.x = 0.12
#        self.att_box.object.poses[0].position.y = 0.0
#        self.att_box.object.poses[0].position.z = 0.0
#        self.att_box.object.poses[0].orientation.x = 0
#        self.att_box.object.poses[0].orientation.y = 0
#        self.att_box.object.poses[0].orientation.z = 0
#        self.att_box.object.poses[0].orientation.w = 1  
#        self.att_box.touch_links = ['r_gripper_palm_link', 'r_gripper_r_finger_link', 'r_gripper_l_finger_link',
#                                    'r_gripper_r_finger_tip_link', 'r_gripper_l_finger_tip_link']
#
#    def testIkWithCollisionObjects(self):
#
#        # add table
#        set_planning_scene_diff_request = SetPlanningSceneDiffRequest()
#        set_planning_scene_diff_request.planning_scene_diff.attached_collision_objects.append(self.att_box)
#
#        self.set_planning_scene_diff.call(set_planning_scene_diff_request)
#
#        kin_req = GetConstraintAwarePositionIKRequest()
#        kin_req.ik_request.ik_link_name = 'r_wrist_roll_link'
#        kin_req.timeout = rospy.Duration(2.0)
#        kin_req.ik_request.pose_stamped.header.frame_id = 'base_link'
#        kin_req.ik_request.pose_stamped.pose.position.x = .52
#        kin_req.ik_request.pose_stamped.pose.position.y = -.2
#        kin_req.ik_request.pose_stamped.pose.position.z = .8
#        kin_req.ik_request.pose_stamped.pose.orientation.x = 0.0
#        kin_req.ik_request.pose_stamped.pose.orientation.y = 0.7071
#        kin_req.ik_request.pose_stamped.pose.orientation.z = 0.0
#        kin_req.ik_request.pose_stamped.pose.orientation.w = 0.7071
#
#        kin_req.ik_request.ik_seed_state.joint_state.name = self.joint_names
#        kin_req.ik_request.ik_seed_state.joint_state.position = [float(0.0) for _ in range(7)]
#
#        kin_req.ik_request.robot_state.joint_state.name = self.joint_names
#        kin_req.ik_request.robot_state.joint_state.position = [float(0.0) for _ in range(7)]
#
#        for _ in range(25):
#
#            while(True):
#                self.state_req.robot_state.joint_state.position = self.generateRandomValues()
#                state_val_res = self.state_validity.call(self.state_req)
#                if(state_val_res.error_code.val == state_val_res.error_code.SUCCESS):
#                    break
#
#            kin_req.ik_request.robot_state.joint_state.position = self.state_req.robot_state.joint_state.position
#            kin_req.ik_request.ik_seed_state.joint_state.position = self.generateRandomValues()
#
#            kin_res = self.ik_service.call(kin_req)
#            
#            self.failIf(kin_res.error_code.val != kin_res.error_code.SUCCESS)
#            self.state_req.robot_state.joint_state.position = kin_res.solution.joint_state.position
#            state_val_res = self.state_validity.call(self.state_req)         
#            self.failIf(state_val_res.error_code.val != state_val_res.error_code.SUCCESS)
                     
if __name__ == '__main__':
    ikt = ik_test()
    #print ikt.results
    output = open('ik_pickle.pkl', 'wb')
    pickle.dump(ikt.results, output, -1)
    output.close()

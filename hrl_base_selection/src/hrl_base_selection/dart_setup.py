import pydart2 as pydart
import numpy as np
import math as m
import copy
import rospkg
import time

import tf.transformations as tft

import roslib, rospkg, rospy
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle
from std_msgs.msg import String
from hrl_msgs.msg import FloatArrayBare
roslib.load_manifest('hrl_base_selection')
from hrl_base_selection.msg import PhysxOutcome
from hrl_base_selection.srv import InitPhysxBodyModel, PhysxInput

from matplotlib.cbook import flatten


class DartDressingWorld(pydart.World):
    def __init__(self, skel_file_name):
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')

        self.human_reference_center_floor_point = None
        # rospy.wait_for_service('init_physx_body_model')
        # self.init_physx_service = rospy.ServiceProxy('init_physx_body_model', InitPhysxBodyModel)
        # rospy.wait_for_service('body_config_input_to_physx')
        # self.update_physx_config_service = rospy.ServiceProxy('body_config_input_to_physx', PhysxInput)


        pydart.World.__init__(self, 1.0 / 2000.0, skel_file_name)
        #print('pydart create_world OK')
        self.ground = self.skeletons[0]

        self.human = self.skeletons[1]
        self.chair = self.skeletons[2]
        self.set_collision_detector(pydart.World.BULLET_COLLISION_DETECTOR)
        #print("detector = %s" % self.collision_detector_string())

        # self.world = pydart.World(1.0 / 2000.0)
        # skel_model = self.add_skeleton(pkg_path + '/models/fullbody1_alex_optimized.skel')
        # pr2_model = self.add_skeleton(pkg_path+'/models/PR2/pr2.urdf')


        self.robot = self.add_skeleton(self.pkg_path+'/models/PR2/pr2.urdf')
        # self.robot = self.add_skeleton('/opt/ros/indigo/share/robot_state_publisher/test/pr2.urdf')
        self.gown_box_leftarm = self.add_skeleton(self.pkg_path + '/models/gown_box_only.urdf')
        self.gown_box_rightarm = self.add_skeleton(self.pkg_path + '/models/gown_box_only.urdf')

        #print('pydart added pr2 OK')
        # self.set_gravity([0.0, 0.0, -9.81])
        # Lock the first joint
        self.robot.joints[0].set_actuator_type(pydart.joint.Joint.LOCKED)

        # Move bit lower (for camera)
        positions = self.robot.positions()
        positions['rootJoint_pos_x'] = 2.
        positions['rootJoint_pos_y'] = 0.
        positions['rootJoint_pos_z'] = 0.
        positions['rootJoint_rot_z'] = 3.14
        self.robot.set_positions(positions)

        q = self.human.q
        q['j_thigh_right_z'] = m.radians(90.)
        q['j_thigh_right_y'] = m.radians(0.)
        q['j_thigh_right_x'] = m.radians(0.)
        q['j_thigh_left_z'] = m.radians(90.)
        q['j_thigh_left_y'] = m.radians(0.)
        q['j_thigh_left_x'] = m.radians(0.)

        q['j_shin_right'] = m.radians(-100.)
        q['j_shin_left'] = m.radians(-100.)

        q['j_heel_right_1'] = m.radians(10.)
        q['j_heel_right_2'] = m.radians(00.)
        q['j_heel_left_1'] = m.radians(10.)
        q['j_heel_left_2'] = m.radians(00.)
        self.human.set_positions(q)

        # self.world_B_gripper = self.robot.bodynode('r_gripper_tool_frame').world_transform()
        # self.gripper_B_gown = np.matrix([[0., -1., 0., 0.03],
        #                                  [1., 0., 0., 0.0],
        #                                  [0., 0., 1., -0.05],
        #                                  [0., 0., 0., 1.0]])
        self.gripper_B_gown = [np.matrix([[0., 1., 0., 0.0303],
                                          [-1., 0., 0., 0.0],
                                          [0., 0., 1., -0.04475],
                                          [0., 0., 0., 1.0]]),
                                np.matrix([[0., -1., 0., 0.0303],
                                          [1., 0., 0., 0.0],
                                          [0., 0., 1., -0.04475],
                                          [0., 0., 0., 1.0]])]
        self.set_gown()

        import xmltodict
        with open(skel_file_name) as fd:
            raw_xml_dict = xmltodict.parse(fd.read())
        self.skel_bodies = raw_xml_dict['skel']['world']['skeleton'][1]['body']
        self.skel_joints = raw_xml_dict['skel']['world']['skeleton'][1]['joint']
        for joint in self.skel_joints:
            if joint['@name'] == 'j_toe_left':
                self.lowest_reference_joint_index = self.skel_joints.index(joint)
            elif joint['@name'] == 'j_pelvis':
                self.center_reference_joint_index = self.skel_joints.index(joint)

        for bodypart in self.skel_bodies:
            if bodypart['@name'] == self.skel_joints[self.lowest_reference_joint_index]['child']:
                self.joint_to_floor_z = float(bodypart['visualization_shape']['geometry']['multi_sphere']['sphere'][0]['radius'])
        self.estimate_center_floor_point()
        positions = self.human.positions()
        positions['j_pelvis1_x'] -= self.human_reference_center_floor_point[0]
        positions['j_pelvis1_y'] -= self.human_reference_center_floor_point[1]
        positions['j_pelvis1_z'] -= self.human_reference_center_floor_point[2]
        self.human.set_positions(positions)
        self.estimate_center_floor_point()

    def displace_gown(self):
        for arm in ['leftarm', 'rightarm']:
            if 'left' in arm:
                flip = 1.
                gown_box = self.gown_box_leftarm
            elif 'right' in arm:
                flip = -1.
                gown_box = self.gown_box_rightarm
            else:
                print 'ERROR'
                print 'I do not know what arm to use!'
            positions = gown_box.positions()
            positions['rootJoint_pos_x'] = flip*5.
            positions['rootJoint_pos_y'] = flip*5.
            positions['rootJoint_pos_z'] = 0.2
            positions['rootJoint_rot_x'] = 0.
            positions['rootJoint_rot_y'] = 0.
            positions['rootJoint_rot_z'] = 0.
            gown_box.set_positions(positions)

    def set_gown(self, arm_selection=None):
        if arm_selection == None:
            arm_selection = ['leftarm', 'rightarm']
        for arm in arm_selection:
            if 'left' in arm:
                gown_box = self.gown_box_leftarm
                i = 1
            elif 'right' in arm:
                gown_box = self.gown_box_rightarm
                i = 0
            else:
                print 'ERROR'
                print 'I do not know what arm to use!'
            self.world_B_gripper = np.matrix(self.robot.bodynode(arm[0]+'_gripper_tool_frame').world_transform())
            # print 'self.world_B_gripper'
            # print self.world_B_gripper
            world_B_gown = self.world_B_gripper*self.gripper_B_gown[i]
            # print 'world_B_gown'
            # print world_B_gown
            positions = gown_box.q
            positions['rootJoint_pos_x'] = world_B_gown[0, 3]
            positions['rootJoint_pos_y'] = world_B_gown[1, 3]
            positions['rootJoint_pos_z'] = world_B_gown[2, 3]

            x_vector = np.array(world_B_gown)[0:3, 0]

            z_origin = np.array([0., 0., 1.])
            y_orth = np.cross(z_origin, x_vector)
            y_orth /= np.linalg.norm(y_orth)
            x_vec_corrected = np.cross(y_orth, z_origin)
            x_vec_corrected /= np.linalg.norm(x_vec_corrected)
            # print x_vector
            # print np.linalg.norm(z_origin)
            # print np.linalg.norm(y_orth)

            world_B_gown_gravity_direction = np.eye(3)
            world_B_gown_gravity_direction[0:3, 0] = copy.copy(x_vec_corrected)
            world_B_gown_gravity_direction[0:3, 1] = copy.copy(y_orth)
            world_B_gown_gravity_direction[0:3, 2] = copy.copy(z_origin)
            # print world_B_gown_gravity_direction

            gown_euler = tft.euler_from_matrix(world_B_gown_gravity_direction, 'sxyz')
            positions['rootJoint_rot_x'] = gown_euler[0]
            positions['rootJoint_rot_y'] = gown_euler[1]
            positions['rootJoint_rot_z'] = gown_euler[2]
            gown_box.set_positions(positions)

    def estimate_center_floor_point(self):
        x_position = self.human.joint(self.skel_joints[self.center_reference_joint_index]['@name']).position_in_world_frame()[0]
        y_position = self.human.joint(self.skel_joints[self.center_reference_joint_index]['@name']).position_in_world_frame()[1]
        z_position = self.human.joint(self.skel_joints[self.lowest_reference_joint_index]['@name']).position_in_world_frame()[2] - self.joint_to_floor_z

        self.human_reference_center_floor_point = np.array([x_position, y_position, z_position])
        #print 'Position of the floor center of the human body with respect to the floor: ', self.human_reference_center_floor_point

    # def on_key_press(self, key):
    #     if key == 'J':
    #         q = self.human.q
    #         q['j_shin_left'] = -2.
    #         q['j_bicep_right_x'] = 1.
    #         q['j_bicep_right_y'] = 0.
    #         q['j_bicep_right_z'] = 2.
    #         q['j_forearm_right_1'] = 0.
    #         q['j_forearm_right_2'] = 0.5
    #
    #
    #         self.human.set_positions(q)
    #         links = []
    #         bodies = []
    #         body_names = []
    #         body_index = []
    #         joint_names = []
    #         spheres = []
    #         for bodypart in self.skel_bodies:
    #             # try:
    #             if 'visualization_shape' in bodypart.keys():
    #                 if 'multi_sphere' in bodypart['visualization_shape']['geometry'].keys():
    #                     multisphere = bodypart['visualization_shape']['geometry']['multi_sphere']['sphere']
    #                     first_sphere_transform = np.eye(4)
    #                     first_sphere_transform[0:3, 3] = np.array([float(t) for t in multisphere[0]['position'].split(' ')])
    #                     transform = np.matrix(self.human.bodynode(bodypart['@name']).world_transform()) * np.matrix(first_sphere_transform)
    #                     position = np.round(copy.copy(np.array(transform)[0:3, 3]) - self.human_reference_center_floor_point, 10)
    #                     radius = float(np.round(float(multisphere[0]['radius']),10))
    #                     sphere_data = copy.copy(list(flatten([position, radius])))
    #                     if sphere_data not in spheres:
    #                         sphere_1_index = len(spheres)
    #                         spheres.append(copy.copy(list(flatten([position, radius]))))
    #                     else:
    #                         sphere_1_index = spheres.index(sphere_data)
    #
    #                     second_sphere_transform = np.eye(4)
    #                     second_sphere_transform[0:3, 3] = np.array([float(t) for t in multisphere[1]['position'].split(' ')])
    #                     transform = np.matrix(self.human.bodynode(bodypart['@name']).world_transform()) * np.matrix(second_sphere_transform)
    #                     position = np.round(copy.copy(np.array(transform)[0:3, 3]) - self.human_reference_center_floor_point, 10)
    #                     radius = float(np.round(float(multisphere[1]['radius']), 10))
    #                     sphere_data = copy.copy(list(flatten([position, radius])))
    #                     if sphere_data not in spheres:
    #                         sphere_2_index = len(spheres)
    #                         spheres.append(copy.copy(list(flatten([position, radius]))))
    #                     else:
    #                         print 'the sphere was already there!!'
    #                         sphere_2_index = spheres.index(sphere_data)
    #                     links.append([sphere_1_index, sphere_2_index])
    #
    #         print 'links'
    #         print links
    #         print 'len(spheres)'
    #         print len(spheres)
    #         # print 'len(body_names)'
    #         # print len(body_names)
    #         print 'len(links)'
    #         print len(links)
    #         spheres = np.array(spheres)
    #         links = np.array(links)
    #         spheres_x = [float(i) for i in spheres[:, 0]]
    #         spheres_y = [float(i) for i in spheres[:, 1]]
    #         spheres_z = [float(i) for i in spheres[:, 2]]
    #         spheres_r = [float(i) for i in spheres[:, 3]]
    #         first_sphere_list = [int(i) for i in links[:, 0]]
    #         second_sphere_list = [int(i) for i in links[:, 1]]
    #         # print spheres_x
    #         # print spheres_y
    #         # print spheres_z
    #         # print spheres_r
    #         # print first_sphere_list
    #         # print second_sphere_list
    #         # spheres_x = [1., 1.5, 2.]
    #         # spheres_y = [1., 1.5, 2.]
    #         # spheres_z = [1., 1.5, 2.]
    #         # spheres_r = [0.1, 0.2, 0.5]
    #         # first_sphere_list = [0, 1]
    #         # second_sphere_list = [1, 2]
    #         resp = self.init_physx_service(spheres_x, spheres_y, spheres_z, spheres_r, first_sphere_list,
    #                                        second_sphere_list)
    #         print 'Physx initialization was successful? ', resp
    #         rospy.sleep(1)
    #         start_traj = [float(i) for i in [-3., 0., 0.]]
    #         end_traj = [float(i) for i in [0., 0., 0.]]
    #         resp = self.update_physx_config_service(spheres_x, spheres_y, spheres_z, spheres_r, first_sphere_list,
    #                                                 second_sphere_list, start_traj, end_traj)
    #         print 'Physx update was successful? ', resp


if __name__ == '__main__':
    rospy.init_node('dart_test_file')
    print('Example: gravity compensation')

    pydart.init()
    print('pydart initialization OK')

    world = DartDressingWorld()

    win = pydart.gui.viewer.PydartWindow(world)
    win.camera_event(1)
    win.set_capture_rate(10)
    win.run_application()


import pydart2 as pydart
import numpy as np
import math as m
import copy
import rospkg
import time


import roslib, rospkg, rospy
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle
from std_msgs.msg import String
from hrl_msgs.msg import FloatArrayBare
roslib.load_manifest('hrl_base_selection')
from hrl_base_selection.helper_functions import createBMatrix, Bmat_to_pos_quat, calc_axis_angle
from hrl_base_selection.msg import PhysxOutcome
from hrl_base_selection.srv import InitPhysxBodyModel, PhysxInput

from matplotlib.cbook import flatten


class GravityCompensationController(object):

    def __init__(self, robot):
        self.robot = robot
        self.g = self.robot.world.gravity()
        self.enabled = True

    def compute(self, ):
        tau = np.zeros(self.robot.num_dofs())
        if not self.enabled:
            return tau

        for body in self.robot.bodynodes:
            m = body.mass()  # Or, simply body.m
            J = body.linear_jacobian(body.local_com())
            tau += J.transpose().dot(-(m * self.g))
        return tau


class MyWorld(pydart.World):

    def __init__(self, ):
        # pydart.World.__init__(self, 0.001)
        self.human_reference_center_floor_point = None
        # rospy.wait_for_service('init_physx_body_model')
        self.init_physx_service = rospy.ServiceProxy('init_physx_body_model', InitPhysxBodyModel)
        # rospy.wait_for_service('body_config_input_to_physx')
        self.update_physx_config_service = rospy.ServiceProxy('body_config_input_to_physx', PhysxInput)
        print('pydart create_world OK')
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')
        # skel_file = '/home/ari/git/catkin_ws/src/hrl-assistive/hrl_base_selection/models/fullbody_alex_capsule.skel'
        skel_file = '/home/ari/git/catkin_ws/src/hrl-assistive/hrl_base_selection/models/fullbody_50percentile_capsule.skel'
        # pydart.World.__init__(self, 1.0 / 2000.0, '/home/ari/git/catkin_ws/src/hrl-assistive/hrl_base_selection/models/world_and_human.skel')
        pydart.World.__init__(self, 1.0 / 2000.0, skel_file)
        # pydart.World.__init__(self, 1.0 / 2000.0, '/home/ari/git/catkin_ws/src/hrl-assistive/hrl_base_selection/models/fullbody1_wenhao_optimized.skel')
        self.human = self.skeletons[1]
        self.set_collision_detector(pydart.World.BULLET_COLLISION_DETECTOR)
        print("detector = %s" % self.collision_detector_string())

        # self.world = pydart.World(1.0 / 2000.0)
        # skel_model = self.add_skeleton(pkg_path + '/models/fullbody1_alex_optimized.skel')
        # pr2_model = self.add_skeleton(pkg_path+'/models/PR2/pr2.urdf')

        self.robot = self.add_skeleton(pkg_path+'/models/PR2/pr2.urdf')
        print('pydart add_skeleton OK')
        self.set_gravity([0.0, 0.0, -9.81])
        # Lock the first joint
        self.robot.joints[0].set_actuator_type(pydart.joint.Joint.LOCKED)

        # Move bit lower (for camera)
        positions = self.robot.positions()
        positions['rootJoint_pos_x'] = 2.75
        positions['rootJoint_pos_y'] = 0.
        positions['rootJoint_pos_z'] = 0.1
        positions['rootJoint_rot_z'] = 3.14
        self.robot.set_positions(positions)

        positions = self.human.positions()
        positions['j_pelvis1_x'] = 0.
        positions['j_pelvis1_y'] = 0.
        positions['j_pelvis1_z'] = 1.
        self.human.set_positions(positions)

        # print self.human.body

        # print 'human self collision check'
        # print self.human.self_collision_check()
        # print 'robot self collision check'
        # print self.robot.self_collision_check()

        # print 'adjacency'
        # print self.human.adjacent_body_check()

        q = self.human.q
        # q['j_shin_left'] = -2.
        # q['j_bicep_right_x'] = 1.
        # q['j_bicep_right_y'] = 0.
        # q['j_bicep_right_z'] = 0.
        # q['j_forearm_right_1'] = 0.
        # q['j_forearm_right_2'] = 0.5
        # q['j_shin_left'] = -2.

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


        q['j_bicep_right_x'] = -1*m.radians(25.0)
        q['j_bicep_right_y'] = m.radians(10.)
        q['j_bicep_right_z'] = m.radians(45.00)
        # q['j_bicep_right_roll'] = m.radians(0.00)
        q['j_forearm_right_1'] = m.radians(00.)
        q['j_forearm_right_2'] = 0.

        q['j_bicep_left_x'] = m.radians(0.0)
        q['j_bicep_left_y'] = -1*m.radians(20.)
        q['j_bicep_left_z'] = m.radians(00.00)
        # q['j_bicep_left_roll'] = -1*m.radians(0.00)
        q['j_forearm_left_1'] = m.radians(45.)
        q['j_forearm_left_2'] = 0.

        self.human.set_positions(q)

        origin_B_hand = np.matrix(self.human.bodynode('h_hand_right').world_transform())
        print 'origin_B_hand'
        print origin_B_hand

        origin_B_upperarm = np.matrix(self.human.bodynode('h_bicep_right').world_transform())
        skeleton_frame_B_worldframe = np.matrix([[1., 0., 0., 0.],
                                                 [0., 0., 1., 0.],
                                                 [0., -1., 0., 0.],
                                                 [0., 0., 0., 1.]])
        origin_B_upperarm_world = origin_B_upperarm * skeleton_frame_B_worldframe
        print 'origin_B_upperarm_world'
        print origin_B_upperarm_world

        v = self.robot.q
        self.robot_arm = 'leftarm'
        v['l_shoulder_pan_joint'] = 2.
        v['r_shoulder_pan_joint'] = -2.
        v[self.robot_arm[0] + '_shoulder_lift_joint'] = 0.
        v[self.robot_arm[0] + '_upper_arm_roll_joint'] = 0.
        v[self.robot_arm[0] + '_elbow_flex_joint'] = 0.
        v[self.robot_arm[0] + '_forearm_roll_joint'] = 0.
        v[self.robot_arm[0] + '_wrist_flex_joint'] = 0.
        v[self.robot_arm[0] + '_wrist_roll_joint'] = 0.
        v['torso_lift_joint'] = 0.3
        self.robot.set_positions(v)

        print 'world collision list'
        print self.human.set_self_collision_check(True)
        self.check_collision()

        # print self.robot.self_collision_check()
        # print 'self.collision_result.contacts'
        # print self.collision_result.contacts
        for contact in self.collision_result.contacts:
            if ((self.robot == contact.skel1 or self.robot == contact.skel2) and
                    (self.robot.bodynode('base_link') == contact.bodynode1 or self.robot.bodynode('base_link') == contact.bodynode2)):
                print contact.skel1
                print contact.bodynode1
                print contact#.skel1, contact.skel2

        arm_parts = [self.human.bodynode('h_bicep_left'),
                     self.human.bodynode('h_forearm_left'),
                     self.human.bodynode('h_hand_left'),
                     self.human.bodynode('h_hand_left2')]
        for contact in self.collision_result.contacts:
            contacts = [contact.bodynode1, contact.bodynode2]
            for arm_part in arm_parts:
                if arm_part in contacts and self.human == contact.skel1 and self.human == contact.skel2:
                    contacts.remove(arm_part)
                    if contacts:
                        if contacts[0] not in arm_parts and not contacts[0] == self.human.bodynode('h_scapula_left'):
                            print contact.skel1
                            print contact.bodynode1
                            print contact  # .skel1, contact.skel2

        # print 'number of contacts:'
        # print len(self.collision_result.contacts)

        # Initialize the controller
        self.controller = GravityCompensationController(self.robot)
        self.robot.set_controller(self.controller)
        print('create controller OK')

        import xmltodict
        with open(skel_file) as fd:
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

    def estimate_center_floor_point(self):
        x_position = self.human.joint(self.skel_joints[self.center_reference_joint_index]['@name']).position_in_world_frame()[0]
        y_position = self.human.joint(self.skel_joints[self.center_reference_joint_index]['@name']).position_in_world_frame()[1]
        z_position = self.human.joint(self.skel_joints[self.lowest_reference_joint_index]['@name']).position_in_world_frame()[2] - self.joint_to_floor_z

        self.human_reference_center_floor_point = np.array([x_position, y_position, z_position])
        # self.human_reference_center_floor_point = np.array([1., 1., 0.])
        print 'Position of the floor center of the human body with respect to the floor: ', self.human_reference_center_floor_point

    def on_key_press(self, key):
        if key == 'G':
            self.controller.enabled = not self.controller.enabled
        if key == 'J':
            q = self.human.q
            # q['j_shin_left'] = -2.
            # q['j_bicep_right_x'] = 1.
            # q['j_bicep_right_y'] = 0.
            # q['j_bicep_right_z'] = 0.
            # q['j_forearm_right_1'] = 0.
            # q['j_forearm_right_2'] = 0.5
            # q['j_shin_left'] = -2.
            q['j_bicep_right_x'] = -1.570796
            q['j_bicep_right_y'] = 1.0
            q['j_bicep_right_z'] = 1.
            q['j_forearm_right_1'] = 1.570796
            q['j_forearm_right_2'] = 0.

            self.human.set_positions(q)
            links = []
            bodies = []
            body_names = []
            body_index = []
            joint_names = []
            spheres = []
            for bodypart in self.skel_bodies:
                # try:
                if 'visualization_shape' in bodypart.keys():
                    if 'multi_sphere' in bodypart['visualization_shape']['geometry'].keys():
                        multisphere = bodypart['visualization_shape']['geometry']['multi_sphere']['sphere']
                        first_sphere_transform = np.eye(4)
                        first_sphere_transform[0:3, 3] = np.array([float(t) for t in multisphere[0]['position'].split(' ')])
                        transform = np.matrix(self.human.bodynode(bodypart['@name']).world_transform()) * np.matrix(first_sphere_transform)
                        position = np.round(copy.copy(np.array(transform)[0:3, 3]) - self.human_reference_center_floor_point, 10)
                        radius = float(np.round(float(multisphere[0]['radius']),10))
                        sphere_data = copy.copy(list(flatten([position, radius])))
                        if sphere_data not in spheres:
                            sphere_1_index = len(spheres)
                            spheres.append(copy.copy(list(flatten([position, radius]))))
                        else:
                            sphere_1_index = spheres.index(sphere_data)

                        second_sphere_transform = np.eye(4)
                        second_sphere_transform[0:3, 3] = np.array([float(t) for t in multisphere[1]['position'].split(' ')])
                        transform = np.matrix(self.human.bodynode(bodypart['@name']).world_transform()) * np.matrix(second_sphere_transform)
                        position = np.round(copy.copy(np.array(transform)[0:3, 3]) - self.human_reference_center_floor_point, 10)
                        radius = float(np.round(float(multisphere[1]['radius']), 10))
                        sphere_data = copy.copy(list(flatten([position, radius])))
                        if sphere_data not in spheres:
                            sphere_2_index = len(spheres)
                            spheres.append(copy.copy(list(flatten([position, radius]))))
                        else:
                            print 'the sphere was already there!!'
                            sphere_2_index = spheres.index(sphere_data)
                        links.append([sphere_1_index, sphere_2_index])

            # for sphere in spheres:
            #     print sphere


                # except:
                #     pass


            '''
            for joint in self.skel_joints:
                child = joint['child']
                parent = joint['parent']
                child_is_multisphere = False
                parent_is_multisphere = False
                for bodypart in self.skel_bodies:
                    if bodypart['@name'] == parent:
                        try:
                            if 'multi_sphere' in bodypart['visualization_shape']['geometry'].keys():
                                # print 'multisphere parent'
                                parent_multisphere = bodypart['visualization_shape']['geometry']['multi_sphere']['sphere']
                                # print
                                # parent_index = self.human.bodynodes.index(bodypart)
                                parent_is_multisphere = True
                                # print 'ok'
                        except:
                            pass
                    elif bodypart['@name'] == child:
                        try:
                            if 'multi_sphere' in bodypart['visualization_shape']['geometry'].keys():
                                # print 'multisphere child'
                                child_multisphere = bodypart['visualization_shape']['geometry']['multi_sphere']['sphere']
                                # child_index = self.human.bodynodes.index(bodypart)
                                child_is_multisphere = True
                                # print 'ok'
                        except:
                            pass
                if child_is_multisphere and parent_is_multisphere:
                    # if parent not in body_names:
                    #     body_names.append(parent)
                    if child not in body_names:
                        body_names.append(child)
                        links.append([body_names.index(parent)+1, body_names.index(child)])
                        second_sphere_transform = np.eye(4)
                        second_sphere_transform[0:3, 3] = np.array([float(t) for t in child_multisphere[1]['position'].split(' ')])
                        transform = np.matrix(self.human.bodynode(child).world_transform()) * np.matrix(second_sphere_transform)
                        position = copy.copy(np.array(transform)[0:3, 3]) - self.human_reference_center_floor_point
                        radius = float(child_multisphere[1]['radius'])
                        spheres.append(copy.copy(list(flatten([position, radius]))))

                elif (child_is_multisphere and not parent_is_multisphere) :
                # elif child_is_multisphere:
                    if child not in body_names:
                        body_names.append(child)
                        joint_names.append(joint['@name'])

                        links.append([joint_names.index(joint['@name']), joint_names.index(joint['@name'])+1])
                        first_sphere_transform = np.eye(4)
                        first_sphere_transform[0:3, 3] = np.array([float(t) for t in child_multisphere[0]['position'].split(' ')])
                        # print child
                        transform = np.matrix(self.human.bodynode(child).world_transform())* np.matrix(
                            first_sphere_transform)
                        position = copy.copy(np.array(transform)[0:3, 3]) - self.human_reference_center_floor_point
                        radius = float(child_multisphere[0]['radius'])
                        # print position
                        # print radius
                        # print list(flatten([position[0:3, 3], radius]))
                        spheres.append(copy.copy(list(flatten([position, radius]))))

                        second_sphere_transform = np.eye(4)
                        second_sphere_transform[0:3, 3] = np.array([float(t) for t in child_multisphere[1]['position'].split(' ')])
                        transform = np.matrix(self.human.bodynode(child).world_transform()) * np.matrix(
                            second_sphere_transform)
                        position = copy.copy(np.array(transform)[0:3, 3]) - self.human_reference_center_floor_point
                        radius = float(child_multisphere[1]['radius'])
                        spheres.append(copy.copy(list(flatten([position, radius]))))
                        # links.append([len(spheres)-2, len(spheres) -1])
                # print 'spheres'
                # print spheres
                '''
            # print 'spheres'
            # print spheres
            # print 'body names'
            # print body_names
            # print 'links'
            # print links
            # print 'len(spheres)'
            # print len(spheres)
            # print 'len(body_names)'
            # print len(body_names)
            # print 'len(links)'
            # print len(links)
            spheres = np.array(spheres)
            links = np.array(links)
            spheres_x = [float(i) for i in spheres[:, 0]]
            spheres_y = [float(i) for i in spheres[:, 1]]
            spheres_z = [float(i) for i in spheres[:, 2]]
            spheres_r = [float(i) for i in spheres[:, 3]]
            first_sphere_list = [int(i) for i in links[:, 0]]
            second_sphere_list = [int(i) for i in links[:, 1]]
            # print spheres_x
            # print spheres_y
            # print spheres_z
            # print spheres_r
            # print first_sphere_list
            # print second_sphere_list
            # spheres_x = [1., 1.5, 2.]
            # spheres_y = [1., 1.5, 2.]
            # spheres_z = [1., 1.5, 2.]
            # spheres_r = [0.1, 0.2, 0.5]
            # first_sphere_list = [0, 1]
            # second_sphere_list = [1, 2]
            resp = self.init_physx_service(spheres_x, spheres_y, spheres_z, spheres_r, first_sphere_list,
                                           second_sphere_list)
            print 'Physx initialization was successful? ', resp
            rospy.sleep(1)

            import tf.transformations as tft
            thx = m.radians(90.)
            x = np.matrix([[1, 0, 0, 0], [0, m.cos(thx), -m.sin(thx), 0], [0, m.sin(thx), m.cos(thx), 0], [0, 0, 0, 1]])
            thz = m.radians(180)
            z = np.matrix([[m.cos(thz), -m.sin(thz), 0 ,0],[m.sin(thz),m.cos(thz),0,0],[0, 0,1,0],[0,0,0,1]])
            thy = m.radians(0.)
            y = np.matrix([[m.cos(thy), 0, m.sin(thy) ,0],[0,1,0,0],[-m.sin(thy), 0,m.cos(thy),0],[0,0,0,1]])
            rot = z*y*x
            quat = tft.quaternion_from_matrix(rot)

            print 'x_vector:', rot[0:3, 0]
            print 'rot'
            print rot
            # liftlink_B_reference = createBMatrix([0., 0., 0.], [0., 0., 0., 1.])

            start_traj = [float(i) for i in [-3., 0., 0.]]
            end_traj = [float(i) for i in [0., 0., 0.]]

            quaternion = quat
            axis, angle = calc_axis_angle(quaternion)
            rospy.sleep(4)
            start_traj = [float(i) for i in [0.1, 0., 0.]]
            end_traj = [float(i) for i in [0.6, 0., 0.]]
            # translation = start_traj
            # resp = self.update_physx_config_service(spheres_x, spheres_y, spheres_z, spheres_r, first_sphere_list,
            #                                         second_sphere_list, start_traj, end_traj, axis, angle)
            # print 'Physx update was successful? ', resp
            # rospy.sleep(4)
            # start_traj = [float(i) for i in [0.3, 0., 0.]]
            # end_traj = [float(i) for i in [0., 0., 0.]]
            # translation = start_traj
            # resp = self.update_physx_config_service(spheres_x, spheres_y, spheres_z, spheres_r, first_sphere_list,
            #                                         second_sphere_list, start_traj, end_traj, axis, angle)
            # print 'Physx update was successful? ', resp

            start_traj = [float(i) for i in [0.8, 0., 1.]]
            end_traj = [float(i) for i in [0.3, 0., 1.1]]
            # translation = start_traj
            resp = self.update_physx_config_service(spheres_x, spheres_y, spheres_z, spheres_r, first_sphere_list,
                                                    second_sphere_list, start_traj, end_traj)
            print 'Physx update was successful? ', resp

    def draw_with_ri(self, ri):
        ri.set_color(0, 0, 0)
        ri.draw_text([20, 40], "time = %.4fs" % self.t)
        ri.draw_text([20, 70], "Gravity Compensation = %s" %
                     ("ON" if self.controller.enabled else "OFF"))


if __name__ == '__main__':
    rospy.init_node('dart_test_file_s')
    print('Example: gravity compensation')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()

    win = pydart.gui.viewer.PydartWindow(world)
    win.camera_event(1)
    win.set_capture_rate(10)
    win.run_application()


#!/usr/bin/env python

import sys
import argparse
import threading

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

from hrl_task_planning import pddl_utils as pddl
from hrl_msgs.msg import FloatArrayBare
from hrl_task_planning.msg import PDDLState

class BedDistanceTracker(object):
    def __init__(self, domain):
        self.domain = domain
        self.frame_lock = threading.RLock()
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        self.state_pub = rospy.Publisher('/move_back_safe_zone', PoseStamped, queue_size=10, latch=True)
        self.model = None
        self.too_close = False
        self.ar_tag_subscriber = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)

    def ar_distance_check(self, pos):
        BED_HARD_THRESH = 1.3
        WHEELCHAIR_X_THRESH = 1.5
        WHEELCHAIR_Y_THRESH = 0.4
        ar_dist_x = pos[0]
        ar_dist_y = pos[1]

        if self.model.upper() == 'AUTOBED':
            if abs(ar_dist_y) < BED_HARD_THRESH:
                preds.append(pddl.Predicate('TOO-CLOSE', [self.model]))
                self.too_close = True
            else:
                preds.append(pddl.Predicate('TOO-CLOSE', [self.model], neg=True))
                self.too_close = False
        elif self.model.upper() == 'WHEELCHAIR':
            if ar_dist_y > WHEELCHAIR_Y_THRESH:
                preds.append(pddl.Predicate('TOO-CLOSE', [self.model], neg=True))
                self.too_close = False
            elif ar_dist_y > 0 and ar_dist_x > WHEELCHAIR_X_THRESH:
                preds.append(pddl.Predicate('TOO-CLOSE', [self.model], neg=True))
                self.too_close = False
            else:
                preds.append(pddl.Predicate('TOO-CLOSE', [self.model]))
                self.too_close = True
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.predicates = map(str, preds)
        self.state_pub.publish(state_msg)
        if self.too_close:

    def arTagCallback(self, msg):
        with self.frame_lock:
            try:
                self.model = rospy.get_param('/pddl_tasks/%s/model_name' % self.domain, 'AUTOBED')
            except KeyError:
                rospy.logwarn("[%s] Tracking AR Tag, but current model unknown! Cannot update PDDLState", rospy.get_name())
                return
            if self.model.lower() == 'autobed':
                self.out_frame = 'autobed/base_link'
                self.config_autobed_AR_detector()
            elif self.mode.lower() == 'wheelchair':
                self.out_frame = 'wheelchair/base_link'
                self.config_wheelchair_AR_detector()
            else:
                print 'I do not know what AR tag to look for... Abort!'
                return

            markers = msg.markers
            for i in xrange(len(markers)):
                cur_p = np.array([markers[i].pose.pose.position.x,
                                  markers[i].pose.pose.position.y,
                                  markers[i].pose.pose.position.z])
                cur_q = np.array([markers[i].pose.pose.orientation.x,
                                  markers[i].pose.pose.orientation.y,
                                  markers[i].pose.pose.orientation.z,
                                  markers[i].pose.pose.orientation.w])
                if len(self.quat_buf) < 1:
                    self.pos_buf.append( cur_p )
                    self.quat_buf.append( cur_q )
                else:
                    first_p = self.pos_buf[0]
                    first_q = self.quat_buf[0]

                    # check close quaternion and inverse
                    if np.dot(cur_q, first_q) < 0.0:
                        cur_q *= -1.0

                    self.pos_buf.append(cur_p)
                    self.quat_buf.append(cur_q)

                positions = self.pos_buf.get_array()
                quaternions = self.quat_buf.get_array()

                pos = None
                quat = None
                if True:
                    # median
                    positions = np.sort(positions, axis=0)
                    pos_int = positions[len(positions)/2-1:len(positions)/2+1]
                    pos = np.sum(pos_int, axis=0)
                    pos /= float(len(pos_int))

                    quaternions = np.sort(quaternions, axis=0)
                    quat_int = quaternions[len(quaternions)/2-1:len(quaternions)/2+1]
                    quat = np.sum(quat_int, axis=0)
                    quat /= float(len(quat_int))
                
                map_B_ar = createBMatrix(pos, quat)
                map_B_ar = self.shift_to_ground(map_B_ar)
                map_B_reference = map_B_ar*self.reference_B_ar.I
                robot_ar_pos = map_B_ref[0:3, 3] 
                self.ar_distance_check(robot_ar_pos)

    def config_wheelchair_AR_detector(self):
        self.tag_id = [13]#[13, 1, 0]  # 9
        self.tag_side_length = 0.11  # 0.053  # 0.033

        # This is the translational transform from reference markers to the bed origin.
        # -.445 if right side of body. .445 if left side.
        model_trans_B_ar_1 = np.eye(4)
        model_trans_B_ar_2 = np.eye(4)
        model_trans_B_ar_3 = np.eye(4)

        # Now that I adjust the AR tag pose to be on the ground plane, no Z shift needed.
        model_trans_B_ar_1[0:3, 3] = np.array([0.30, 0.00, 0.])
        #model_trans_B_ar_2[0:3, 3] = np.array([-0.03, 0.02, 0.])
        #model_trans_B_ar_3[0:3, 3] = np.array([-0.03, 0.02, 0.])

        #ar_roty_B[0:3, 0:3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        # If left side of bed should be np.array([[-1,0],[0,-1]])
        self.reference_B_ar = np.matrix(model_trans_B_ar_1)
        #self.reference_B_ar_2 = np.matrix(model_trans_B_ar_2)
        #self.reference_B_ar_3 = np.matrix(model_trans_B_ar_3)

    def config_autobed_AR_detector(self):
        self.tag_id = [4]  # 9

        # self.autobed_sub = rospy.Subscriber('/abdout0', FloatArrayBare, self.bed_state_cb)
        self.tag_side_length = 0.15  # 0.053  # 0.033

        # This is the translational transform from reference markers to the bed origin.
        # -.445 if right side of body. .445 if left side.
        model_trans_B_ar = np.eye(4)
        # model_trans_B_ar[0:3, 3] = np.array([-0.01, .00, 1.397])
        # Now that I adjust the AR tag pose to be on the ground plane, no Z shift needed.
        model_trans_B_ar[0:3, 3] = np.array([-0.03, -0.05, 0.])
        ar_rotz_B = np.eye(4)
        #ar_rotz_B[0:2, 0:2] = np.array([[-1, 0], [0, -1]])

        ar_roty_B = np.eye(4)
        #ar_roty_B[0:3, 0:3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        # If left side of bed should be np.array([[-1,0],[0,-1]])
        # If right side of bed should be np.array([[1,0],[0,1]])
        ar_rotx_B = np.eye(4)
        # ar_rotx_B[1:3, 1:3] = np.array([[0,1],[-1,0]])
        self.reference_B_ar = np.matrix(model_trans_B_ar)*np.matrix(ar_roty_B)*np.matrix(ar_rotz_B)


    def shift_to_ground(self, this_map_B_ar):
        with self.frame_lock:
            ar_rotz_B = np.eye(4)
            ar_rotz_B[0:2, 0:2] = np.array([[-1, 0], [0, -1]])

            ar_roty_B = np.eye(4)
            ar_roty_B[0:3, 0:3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            ar_rotx_B = np.eye(4)
            orig_B_properly_oriented = np.matrix(ar_roty_B)*np.matrix(ar_rotz_B)

            this_map_B_ar = this_map_B_ar*orig_B_properly_oriented.I

            z_origin = np.array([0, 0, 1])
            x_bed = np.array([this_map_B_ar[0, 0], this_map_B_ar[1, 0], this_map_B_ar[2, 0]])
            y_bed_project = np.cross(z_origin, x_bed)
            y_bed_project = y_bed_project/np.linalg.norm(y_bed_project)
            x_bed_project = np.cross(y_bed_project, z_origin)
            x_bed_project = x_bed_project/np.linalg.norm(x_bed_project)
            map_B_ar_project = np.eye(4)
            for i in xrange(3):
                map_B_ar_project[i, 0] = x_bed_project[i]
                map_B_ar_project[i, 1] = y_bed_project[i]
                map_B_ar_project[i, 3] = this_map_B_ar[i, 3]
            map_B_ar_floor = copy.deepcopy(np.matrix(map_B_ar_project))
            map_B_ar_floor[2, 3] = 0.
            return map_B_ar_floor

	
def main():
    rospy.init_node('bed_distance_monitor')
    parser = argparse.ArgumentParser(description="Report when bed is too-close")
    parser.add_argument('--domain', '-d', help="The domain this monitor is updating.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = BedDistanceTracker(args.domain)
    rospy.spin()

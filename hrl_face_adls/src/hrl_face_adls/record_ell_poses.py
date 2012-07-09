#!/usr/bin/python

import sys
import yaml
import numpy as np

import roslib
roslib.load_manifest("hrl_face_adls")
import rospy

from hrl_pr2_arms.pr2_arm import create_pr2_arm
from hrl_ellipsoidal_control.ellipsoid_controller import EllipsoidParamServer
from hrl_face_adls.srv import RequestRegistration

def main():
    rospy.init_node("record_ell_poses")
    assert len(sys.argv) == 4
    tool = sys.argv[1]
    modes = {"shaver" : "shaving", "spoon" : "feeding", "scratcher" : "scratching" }
    mode = modes[tool]
    side = sys.argv[2]

    request_registration = rospy.ServiceProxy("/request_registration", RequestRegistration)
    print "Waiting for /request_registration"
    request_registration.wait_for_service()
    raw_input("Load Registration")
    reg_resp = request_registration(mode, side)
    if not reg_resp.success:
        print "Not registered"
        return
    eps = EllipsoidParamServer()
    eps.load_params(reg_resp.e_params)
    end_link = "l_gripper_%s45_frame" % tool
    arm = create_pr2_arm('l', end_link=end_link, timeout=1.)
    poses = {}
    while not rospy.is_shutdown():
        name = raw_input("Type name for this pose (Enter nothing to stop): ")
        if name == "":
            break
        ell_coords, ell_quat = eps.get_ell_pose(arm.get_end_effector_pose())
        poses[name] = [np.array(ell_coords).tolist(), ell_quat.tolist()]
    f = file(sys.argv[3], 'w')
    yaml.dump(poses, f)
    f.close()

if __name__ == "__main__":
    main()

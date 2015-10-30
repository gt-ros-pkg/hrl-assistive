#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import numpy as np
import copy

from ar_track_alvar.msg import AlvarMarkers
import tf
import geometry_msgs
from tf_conversions import posemath
import PyKDL

def poseStamped2transformStamped(ps, ts):
    ts.header = copy.deepcopy(ps.header)
    pose2transform(ps.pose, ts.transform)

def pose2transform(pose, transform):

    transform.translation.x = pose.position.x
    transform.translation.y = pose.position.y
    transform.translation.z = pose.position.z

    transform.rotation.x = pose.orientation.x
    transform.rotation.y = pose.orientation.y
    transform.rotation.z = pose.orientation.z
    transform.rotation.w = pose.orientation.w
    

if __name__ == '__main__':


    ps1 = geometry_msgs.msg.PoseStamped()
    ps1.header.stamp    = rospy.Time(0)
    ps1.header.frame_id = "base_link"
    
    ps1.pose.position.x    = 1.0
    ps1.pose.position.y    = 0.0
    ps1.pose.position.z    = 0.0
    
    ps1.pose.orientation.x = 0.0
    ps1.pose.orientation.y = 0.0
    ps1.pose.orientation.z = 0.0
    ps1.pose.orientation.w = 1.0


    ps2 = geometry_msgs.msg.PoseStamped()
    ps2.header.stamp    = rospy.Time(0)
    ps2.header.frame_id = "base_link"
    
    ps2.pose.position.x    = 0.0
    ps2.pose.position.y    = 1.0
    ps2.pose.position.z    = 0.0
    
    ps2.pose.orientation.x = 0.0
    ps2.pose.orientation.y = 0.7071
    ps2.pose.orientation.z = 0.0
    ps2.pose.orientation.w = 0.7071


    f1 = posemath.fromMsg(ps1.pose)
    f2 = posemath.fromMsg(ps2.pose)
    
    f3 = f1.Inverse()*f2

    p1 = PyKDL.Vector(1,0,0)
    p2 = PyKDL.Vector(2,0,0)


    n_tags = 1
    corner1 = PyKDL.Vector(-1.65, -1.65, 0.0)
    corner2 = PyKDL.Vector( 1.65, -1.65, 0.0)
    corner3 = PyKDL.Vector( 1.65,  1.65, 0.0)
    corner4 = PyKDL.Vector(-1.65,  1.65, 0.0)

    bundle_list = [1]
    bundle_dict = {}
    bundle_dict['1'] = [corner1, corner2, corner3, corner4]
    
    ## f=open('./test.xml', 'w+')

    ## f.write('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n')
    ## f.write('<multimarker markers="'+str(n_tags)+'">\n')

    ## for i in bundle_list:
    ##     d = bundle_dict[str(i)]

    ##     f.write('    <marker index="'+str(i)+'" status="1">\n')
    ##     f.write('        <corner x="'+str(d[0][0])+'" y="'+str(d[0][1])+'" z="'+str(d[0][2])+'" />\n')
    ##     f.write('        <corner x="'+str(d[1][0])+'" y="'+str(d[1][1])+'" z="'+str(d[1][2])+'" />\n')
    ##     f.write('        <corner x="'+str(d[2][0])+'" y="'+str(d[2][1])+'" z="'+str(d[2][2])+'" />\n')
    ##     f.write('        <corner x="'+str(d[3][0])+'" y="'+str(d[3][1])+'" z="'+str(d[3][2])+'" />\n')
    ##     f.write('    </marker>\n')

    ## f.write('</multimarker>\n')
    ## f.close()

    print PyKDL.Frame.Identity()

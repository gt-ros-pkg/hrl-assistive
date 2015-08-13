#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import numpy as np

from ar_track_alvar.msg import AlvarMarkers
import tf
import geometry_msgs

class arTagBundle:

    def __init__(self, main_id, tag_shape):

        self.main_id = main_id
        self.tag_shape= tag_shape
        
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.arTagCallback)


    def arTagCallback(self, msg):

        markers = msg.markers

        main_tag_flag = False
        main_tag_tf   = None
        
        for i in xrange(len(markers)):
            marker_id = markers[i].id

            if marker_id == self.main_id:
                main_tag_flag = True
                main_tag_tf = tf.Transformer(True, rospy.Duration(10.0))
                m = geometry_msgs.msg.TransformStamped()
                m.header.frame_id = markers[i].id
                m.parent_id = markers[i].header.frame_id
                main_tag_tf.setTransform(m)
                
        if main_tag_flag == False: return
                    
        for i in xrange(len(markers)):

            if markers[i].id != self.main_id:                

                tag_tf = tf.Transformer(True, rospy.Duration(10.0))
                m = geometry_msgs.msg.TransformStamped()
                m.header.frame_id = markers[i].id
                m.parent_id = markers[i].header.frame_id
                tag_tf.setTransform(m)

                rel_tf = tag_tf.inverseTimes(main_tag_tf)

                
                
                
                pose = markers[i].pose.pose
                print i, markers[i].id, pose.position
        


if __name__ == '__main__':
    rospy.init_node('ar_tag_bundle_estimation')

    main_id = 9
    tag10 = np.array([0, 0, 0])

    tag_shape = np.hstack([ np.array([-1.65, -1.65, 0.0]).T, np.array([1.65, -1.65, 0.0]).T, \
                            np.array([1.65, 1.65, 0.0]).T, np.array([-1.65, 1.65, 0.0]).T])
        
    atb = arTagBundle(main_id, tag_shape)
    
    rate = rospy.Rate(10) # 25Hz, nominally.    
    while not rospy.is_shutdown():
        ## log.log_state()
        rate.sleep()
    
    

    ## tag20 = np.array([0, 0, 0])
    ## tag21 = np.array([-1.65, -1.65, 0.0])
    ## tag22 = np.array([1.65, -1.65, 0.0])
    ## tag23 = np.array([1.65, 1.65, 0.0])
    ## tag24 = np.array([-1.65, 1.65, 0.0])

    ## tag30 = np.array([0, 0, 0])
    ## tag31 = np.array([-1.65, -1.65, 0.0])
    ## tag32 = np.array([1.65, -1.65, 0.0])
    ## tag33 = np.array([1.65, 1.65, 0.0])
    ## tag34 = np.array([-1.65, 1.65, 0.0])


    ## rot_12_x = np.array([])
    

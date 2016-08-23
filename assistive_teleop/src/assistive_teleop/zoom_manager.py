#!/usr/bin/env python

import rospy

from dynamic_reconfigure import client

from assistive_teleop.srv import SetRegionOfInterest


def VideoZoomManager(object):
    def __init__(self, dynparam_node):
        self.dynparam_client = client.Client(dynparam_node)
        self.paramService = rospy.Service('change_roi', SetRegionOfInterest, self.setROI_cb)

    def setROI_cb(self, roi_req):
        config = {}
        config['width'] = roi_req.width
        config['height'] = roi_req.height
        config['x_offset'] = roi_req.x_offset
        config['y_offset'] = roi_req.y_offset
        if roi_req.width > 960 and roi_req.height >= 540:
            config['decimation_x'] = 2
            config['decimation_y'] = 2
        else:
            config['decimation_x'] = 1
            config['decimation_y'] = 1
        return self.dynparam_client.update_configuration(config)


def main():
    rospy.init_node('video_zoom_manager')
    zoom_manager = VideoZoomManager('/kinect_crop_decimate')
    rospy.spin()

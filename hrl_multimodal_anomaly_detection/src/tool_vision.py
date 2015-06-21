#!/usr/bin/env python

import rospy
from threading import Thread
from visionTracker import visionTracker

class tool_vision(Thread):
    def __init__(self):
        super(tool_vision, self).__init__()
        self.daemon = True
        self.cancelled = False

        self.init_time = 0.

        self.time_data = []
        # A set of 3D points
        self.visual_points = []

        self.visionTracker = visionTracker(useARTags=True, shouldSpin=False, visual=False)

    def reset(self):
        pass

    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""
        rate = rospy.Rate(1000) # 25Hz, nominally.
        while not self.cancelled:
            self.log()
            rate.sleep()

    def log(self):
        point = self.visionTracker.getLogData()
        if point is not None:
            self.visual_points.append(point)
            self.time_data.append(rospy.get_time() - self.init_time)

    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        rospy.sleep(1.0)

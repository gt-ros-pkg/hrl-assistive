#!/usr/bin/env python

__author__ = 'zerickson'

from hrl_multimodal_anomaly_detection.src import visionTracker

if __name__ == '__main__':
    # visionTracker(useARTags=False, targetFrame='/camera_link', shouldSpin=True, publish=True, visual=True)
    visionTracker(useARTags=False, targetFrame='/torso_lift_link', shouldSpin=True, publish=False, visual=True)
    # visionTracker(useARTags=False, targetFrame='/torso_lift_link', shouldSpin=True, publish=True, visual=False)

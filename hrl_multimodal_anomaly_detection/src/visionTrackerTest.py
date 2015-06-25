#!/usr/bin/env python

__author__ = 'zerickson'

from visionTracker import visionTracker

if __name__ == '__main__':
    # visionTracker(useARTags=False, targetFrame='/camera_link', shouldSpin=True, publish=True, visual=True)
    visionTracker(useARTags=False, targetFrame='/torso_lift_link', shouldSpin=True, publish=True, visual=True)
    # visionTracker(useARTags=False, targetFrame='/torso_lift_link', shouldSpin=True, publish=True, visual=False)

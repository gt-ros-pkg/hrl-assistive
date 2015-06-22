#!/usr/bin/env python

__author__ = 'zerickson'

from visionTracker import visionTracker

if __name__ == '__main__':
    visionTracker(useARTags=False, targetFrame='/camera_link', shouldSpin=True, visual=True)

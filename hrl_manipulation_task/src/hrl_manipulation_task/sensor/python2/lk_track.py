#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock
import sys
from sklearn.cluster import KMeans
import copy

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

color_list = [[0,0,255],
              [255,0,0],
              [0,255,0],
              [255,255,255]]


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0

        self.nCluster = 3
        self.km = KMeans(self.nCluster)
        
    def run(self):
        prevPts = None
        currPts = None
        
        while True:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                #d = abs(p0-p1).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                self.tracks = new_tracks


                #clustering by pos and vel
                if prevPts is None:
                    prevPts = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 2)
                else:
                    currPts = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 2)

                    ## if len(prevPts) > len(currPts):                        
                    ##     deltaPts = prevPts[:len(currPts)]-currPts
                    ## else:
                    ##     deltaPts = prevPts-currPts[:len(prevPts)]
                    ##     prevPts = copy.copy(currPts)
                    ##     currPts = currPts[:len(deltaPts)]

                    idx_list = self.km.fit_predict(currPts)
                    max_idx = max(idx_list)

                    for ii, idx in enumerate(idx_list):
                        x = currPts[ii,0]
                        y = currPts[ii,1]
                        cv2.circle(vis, (x, y), 4, color_list[idx], -1) 

                        
                    ## for i in xrange(len(deltaPts)):

                
                ## cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                                          
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = 0

    print __doc__
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

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
import random



lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

## def centroid_histogram(clt):
##     # grab the number of different clusters and create a histogram
##     # based on the number of pixels assigned to each cluster
##     numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
##     (hist, _) = np.histogram(clt.labels_, bins = numLabels)

##      # normalize the histogram, such that it sums to one
##      hist = hist.astype("float")
##      hist /= hist.sum()

##      # return the histogram
##      return hist

## def plot_colors(hist, centroids):
##     # initialize the bar chart representing the relative frequency
##     # of each of the colors
##     bar = np.zeros((50, 300, 3), dtype = "uint8")
##     startX = 0

##     # loop over the percentage of each cluster and the color of
##     # each cluster
##     for (percent, color) in zip(hist, centroids):
##         # plot the relative percentage of each cluster
##         endX = startX + (percent * 300)
##         cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
##                       color.astype("uint8").tolist(), -1)
##         startX = endX

##     # return the bar chart
##     return bar

## def getCentroidColor(clt):
##     # grab the number of different clusters and create a histogram
##     # based on the number of pixels assigned to each cluster
##     numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    
##     label = clt.labels_


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0

        self.nCluster = 6
        self.km = KMeans(self.nCluster)
        self.cluster_centers = []
        self.id = {} # id, center, current label
        

        self.color_list = [[0,0,255],
                      [255,0,0],
                      [0,255,0],
                      [255,255,255]]
        for i in xrange(self.nCluster-4):
            self.color_list.append([random.randint(0,255),
                                    random.randint(0,255),
                                    random.randint(0,255) ])
            

        

        
    def run(self):
        prevPts = None
        currPts = None
        
        while True:
            ret, frame = self.cam.read()
            frame_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                #d = abs(p0-p1).reshape(-1, 2).max(-1)
                good = []
                for v in d:
                    if v < 1: # and v > 0.0001:
                        good.append(True)
                    else:
                        good.append(False)                    
                    ## good = (d < 1) * (d>0.01)

                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                self.tracks = new_tracks


                #clustering by color, pos, and vel
                ## rgbPts = []
                ## for tr in self.tracks:
                ##     [x,y]  = tr[-1]
                ##     x = int(y)
                ##     y = int(x)
                ##     rgbPts.append([np.float32(frame_color[x,y,0]), np.float32(frame_color[x,y,1]), \
                ##                   np.float32(frame_color[x,y,2])] )
                ## rgbPts = np.array(rgbPts)
                    
                currPts  = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 2)
                deltaPts = np.float32([np.array(tr[-2]) - np.array(tr[-1]) if len(tr)>1 else np.array([0,0]) for tr in self.tracks]).reshape(-1, 2)

                if len(currPts) < self.nCluster: continue

                points = np.hstack([currPts, deltaPts])
                point_max = np.amax(points, axis=0)
                point_min = np.amin(points, axis=0)
                
                points = (points - point_min)/(point_max - point_min)
                labels = self.km.fit_predict(points)
                max_idx = max(labels)


                self.cluster_centers.append(self.km.cluster_centers_)
                clt = KMeans(n_clusters = self.nCluster)
                cluster_seq = np.array(self.cluster_centers).reshape( (np.shape(self.cluster_centers)[0]*\
                                                                        np.shape(self.cluster_centers)[1],4) )
                clt.fit(cluster_seq)
                                                                        
                print np.shape(cluster_seq)
                
                for ii, center in enumerate(self.km.cluster_centers_):
                    x = int(center[0]*(point_max[0] - point_min[0]) + point_min[0])
                    y = int(center[1]*(point_max[1] - point_min[1]) + point_min[1])
                    cv2.circle(vis, (x, y), 8, (255,0,0), -1)



                ## if self.id == {}:
                ##     for ii, cur_center in enumerate(self.km.cluster_centers_):
                ##         self.id[ii] = {'center': cur_center,
                ##                        'cur_label': ii }
                ## else:
                ##     remove_id = []
                ##     add_or_keep_id = []
                ##     for ii, cur_center in enumerate(self.km.cluster_centers_):
                ##         new_flag = True
                ##         min_dist = 1000
                ##         min_key  = 0
                ##         for key in self.id.keys():
                ##             ## print np.linalg.norm(cur_center[:2]-self.id[key]['center'][:2])
                ##             dist = np.linalg.norm(cur_center[:2]-self.id[key]['center'][:2])
                ##             if min_dist > dist:
                ##                 min_dist = dist
                ##                 min_key  = key

                ##         if min_dist > 0.5:
                ##             new_id = max(self.id.keys())+1
                ##             self.id[new_id] = {'center': cur_center,
                ##                                'cur_label': ii }
                ##             add_or_keep_id.append(new_id)
                ##         else:
                ##             self.id[key]['center']    = cur_center
                ##             self.id[key]['cur_label'] = ii                            
                ##             add_or_keep_id.append(key)                        

                ##     for key in self.id.keys():
                ##         if key not in add_or_keep_id:
                ##             del self.id[key]


                ## for ii, label in enumerate(labels):
                ##     x = currPts[ii,0]
                ##     y = currPts[ii,1]
                ##     color_idx = 0

                ##     for key in self.id.keys():
                ##         if self.id[key]['cur_label'] == label:
                ##             color_idx = key if key < len(self.color_list) else key%len(self.color_list)
                ##             break

                ##     cv2.circle(vis, (x, y), 4, self.color_list[color_idx], -1)


                # cluster the pixel intensities
                ## image = frame_color.reshape((np.shape(frame_color)[0] * np.shape(frame_color)[1], 3))
                ## clt   = KMeans(n_clusters = 5)
                ## clt.fit(image)

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
            ## cv2.imshow('segmentation', )

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

#!/usr/bin/env python

'''
Camshift tracker
================

This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)

http://www.robinhewitt.com/research/track/camshift.html

Usage:
------
    camshift.py [<video source>]

    To initialize tracking, select the object with mouse

Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

import numpy as np
import cv2
import video
from sklearn.cluster import KMeans
from scipy import stats
import copy

class App(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src)
        ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')
        ## cv2.setMouseCallback('camshift', self.onmouse)

        ## self.selections = []
        self.track_windows = []
        self.hists = []
        
        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.show_backproj = False

        self.init_params()
        self.frame = cv2.resize(self.frame, (0,0), fx=self.scale, fy=self.scale) 


    def init_params(self):

        self.scale      = 0.3
        self.flow_thres = 3.0
        self.n_clusters = 5

        self.last_center = None
        return

    def get_flow_centers(self, flow):

        window_h = 20
        window_w = 20

        h, w = flow.shape[:2]
        yy, xx     = np.meshgrid(range(w), range(h))
        flow_array = flow.reshape((h*w,2))
        mag_array  = np.linalg.norm(flow_array, axis=1)

        data = np.vstack([yy.ravel(), xx.ravel(), mag_array]).T
        flow_filt = data[data[:,2]>self.flow_thres]

        if len(flow_filt) < self.n_clusters: return []

        if self.last_center is not None:
            clt = KMeans(n_clusters = self.n_clusters, init=self.last_center)
        else:
            clt = KMeans(n_clusters = self.n_clusters)
        clt.fit(flow_filt)
        self.last_center = clt.cluster_centers_

        #----------------------------------------------------------
        # flow center
        #----------------------------------------------------------
        new_selections = []
        for ii, center in enumerate(clt.cluster_centers_):
            # loc
            flow_cluster = flow_filt[clt.labels_ == ii]
            l1 = np.amin(flow_cluster[:,:2], axis=0).astype(int)
            r1 = np.amax(flow_cluster[:,:2], axis=0).astype(int)

            # size check
            if r1[0]-l1[0] < 2 or r1[0]-l1[0] > 40: continue
            if r1[1]-l1[1] < 2 or r1[1]-l1[1] > 40: continue
            ## ## if r1[0]-l1[0] < 20 or r1[0]-l1[0] > 40: continue
            ## ## if r1[1]-l1[1] < 20 or r1[1]-l1[1] > 40: continue
            if l1[0] == 0 and r1[0] == 0: continue
            if l1[1] == 0 and r1[1] == 0: continue

            overlap_flag = False
            for jj, window in enumerate(self.track_windows):
                x0, y0, xd, yd = window
                x1 = x0+xd
                y1 = y0+yd

                if not(r1[0] < x0 or x1 < l1[0] or r1[1] < y0 or y1 < l1[1]):
                    overlap_flag = True
                    break
                    

            if overlap_flag is False:
                new_selections.append([l1[0], l1[1], r1[0], r1[1]])

        ##     sub_flow_array = flow_array[data[:,2]>self.flow_thres][clt.labels_ == ii]
        ##     flow_mean = np.mean(, axis=0)
        return new_selections


    def reduce_track_windows(self):

        track_windows = copy.copy(self.track_windows)
        hists = copy.copy(self.hists)

        self.track_windows = []
        self.hists = []

        keep_idx = []
        while True:
            
            window = track_windows[0]
            hist   = hists[0]

            x0, y0, xd, yd = window
            x1 = x0+xd
            y1 = y0+yd

            overlap_idx = []
            non_overlap_idx = []
            for ii, (window2, hist2) in enumerate(zip(track_windows, hists)):
                if ii == 0:
                    overlap_idx.append(ii)
                    continue
                
                x2, y2, xd, yd = window2
                x3 = x2+xd
                y3 = y2+yd

                if not(x1 < x2 or x3 < x0 or y1 < y2 or y3 < y0):
                    ## if np.linalg.norm(hist/np.sum(hist) - hist2/np.sum(hist2)) > 0.5:
                    overlap_idx.append(ii)
                else:
                    non_overlap_idx.append(ii)

            self.track_windows.append(window)
            self.hists.append(hist)
            
            if len(non_overlap_idx) == 0: break
            track_windows = [track_windows[i] for i in non_overlap_idx]
            hists = [hists[i] for i in non_overlap_idx]            

    
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        
        ## print event, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0
        if self.drag_start:
            ## print flags & cv2.EVENT_FLAG_LBUTTON
            ## if flags & cv2.EVENT_FLAG_LBUTTON:
            if flags == 33:
                h, w = self.frame.shape[:2]
                xo, yo = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))
                self.selection = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.selection = (x0, y0, x1, y1)
            ## else:
        if event == cv2.EVENT_LBUTTONUP:            
            self.drag_start = None
        if self.selection is not None and self.drag_start is None:
            self.drag_start = None
            self.tracking_state = 1

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    def run(self):

        # Last gray image
        ret, prev = self.cam.read()
        prev     = cv2.resize(prev, (0,0), fx=self.scale, fy=self.scale) 
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        
        counter = 0
        while True:
            ret, self.frame = self.cam.read()
            self.frame = cv2.resize(self.frame, (0,0), fx=self.scale, fy=self.scale)             
            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            # Current gray image
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)            
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
            prevgray = gray

            # Automatic tracking center selection
            new_selections = self.get_flow_centers(flow)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            
            if len(new_selections)>0:
                for selection in new_selections:
                    x0, y0, x1, y1 = selection
                    hsv_roi = hsv[y0:y1, x0:x1]
                    mask_roi = mask[y0:y1, x0:x1]
                    ## hsv_roi = hsv[y0:y1, x0:x1]
                    ## mask_roi = mask[y0:y1, x0:x1]
                    hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
                    
                    self.hists.append(hist.reshape(-1))
                    ## self.show_hist()

                    ## vis_roi = vis[y0:y1, x0:x1]
                    ## cv2.bitwise_not(vis_roi, vis_roi)
                    ## vis[mask == 0] = 0
                    ## self.selections.append(selection)
                    if x0==0 and y0==0: continue
                    self.track_windows.append((x0, y0, x1-x0, y1-y0))

            if len(self.track_windows)>1: # self.tracking_state == 1:
                ## self.selection = None
                track_boxes = []
                for ii, hist in enumerate(self.hists):
                    prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
                    prob &= mask
                    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

                    if self.track_windows[ii][0]==0 and self.track_windows[ii][1]==0: continue
                    track_box, self.track_windows[ii] = cv2.CamShift(prob, self.track_windows[ii], term_crit)
                    track_boxes.append(track_box)

                self.reduce_track_windows()
                for track_box in track_boxes:

                    ## if self.show_backproj:
                    ##     vis[:] = prob[...,np.newaxis]
                    try: cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                    except: print track_box

                        

                counter += 1
            print "---------------------------------------"
            cv2.imshow('camshift', vis)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    try: video_src = sys.argv[1]
    except: video_src = 0
    print __doc__
    App(video_src).run()

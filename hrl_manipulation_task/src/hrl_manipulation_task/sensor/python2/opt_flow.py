#!/usr/bin/env python

import numpy as np
import cv2
import video
from sklearn.cluster import KMeans
import time, random

help_message = '''
USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

'''
init_center = None
last_center = None
last_label  = None
color_list = [[0,0,255],
              [255,0,0],
              [0,255,0],
              [255,255,255]]
for i in xrange(10):
    color_list.append([random.randint(0,255),
                       random.randint(0,255),
                       random.randint(0,255) ])
cluster_centers = None

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_clustered_flow(img, flow, counter):
    global last_center, last_label, color_list
    global init_center
    global cluster_centers
    
    
    h, w = flow.shape[:2]

    ## start = time.clock()
    yy, xx = np.meshgrid(range(w), range(h))
    flow_array = flow.reshape((h*w,2))
    mag_array  = np.linalg.norm(flow_array, axis=1)

    data = np.vstack([xx.ravel(), yy.ravel(), mag_array]).T
    flow_filt = data[data[:,2]>3.0]
    ## end = time.clock()
    ## print "%.2gs" % (end-start)

    n_clusters = 3
    if len(flow_filt) < n_clusters: return img

    if last_center is not None:
        clt = KMeans(n_clusters = n_clusters, init=init_center)
    else:
        clt = KMeans(n_clusters = n_clusters)
    clt.fit(flow_filt)
    init_center = clt.cluster_centers_
    
    #----------------------------------------------------------
    time_array = np.ones((n_clusters, 1))*counter
    if cluster_centers is None:
        cluster_centers = clt.cluster_centers_
        ## cluster_centers = np.hstack([time_array, clt.cluster_centers_])
    else:
        cluster_centers = np.vstack([ cluster_centers, clt.cluster_centers_ ])
        ## cluster_centers = np.vstack([ cluster_centers, np.hstack([time_array, clt.cluster_centers_]) ])

    if len(cluster_centers) > n_clusters*20:
        cluster_centers = cluster_centers[-n_clusters*20:]

    clt2 = KMeans(n_clusters = n_clusters)
    clt2.fit(cluster_centers)

    if last_label is None: 
        last_center = clt.cluster_centers_[-n_clusters:].tolist()
        last_label  = clt.labels_[-n_clusters:].tolist()
        print "JJJJJJJJJJJJJJJJump"
        return img
    cur_centers = clt.cluster_centers_[-n_clusters:]
    cur_labels  = clt.labels_[-n_clusters:]

    max_label = max(last_label)
    for ii, (center, label) in enumerate(zip(cur_centers, cur_labels)):
        min_dist = 1000
        min_label= 0
        min_idx  = 0
        for jj, (c, l) in enumerate(zip(last_center, last_label)):
            dist = np.linalg.norm(center-c)
            if dist < min_dist:
                min_dist = dist
                min_label= l
                min_idx  = jj

        # new label
        if min_dist > 300:
            cur_labels[ii] = max_label+1
            max_label += 1
        else:
            del last_center[min_idx]
            del last_label[min_idx]
            cur_labels[ii] = min_label

    ######################### Update last centers and labels             
    last_center = cur_centers.tolist()
    last_label  = cur_labels.tolist()

    # cluster center
    overlay = img.copy()
    for ii, (center, label) in enumerate(zip(cur_centers, cur_labels)):
        x = int(center[1])
        y = int(center[0])
        color_idx = label if label < len(color_list) else label%len(color_list)
        c = color_list[color_idx]
        cv2.circle(overlay, (x, y), 8, (c[0], c[1], c[2]), -1)
        ## cv2.circle(overlay, (x, y), 8, (c[0], c[1], int(c[2]*center[2])), -1)

    opacity = 0.5
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # moving direction
    vis = img.copy()
    for ii, center in enumerate(cur_centers):
        x = int(center[1])
        y = int(center[0])
        flow = np.sum(flow_array[data[:,2]>3.0][clt.labels_ == ii], axis=0)
        lines = np.vstack([x, y, x+flow[0], y+flow[1]]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)        
        cv2.polylines(vis, lines, 0, (0, 255, 0))
    
    opacity = 1.0
    cv2.addWeighted(vis, opacity, img, 1 - opacity, 0, img)

        
    return img

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    print help_message
    try: fn = sys.argv[1]
    except: fn = 0

    scale = 0.5
    cam = video.create_capture(fn)
    ret, prev = cam.read()
    prev = cv2.resize(prev, (0,0), fx=scale, fy=scale) 
    
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    ## video = cv2.VideoWriter("temp.avi", -1, 1, (np.shape(prev)[0],np.shape(prev)[1]) )
    counter = 0
    while True:
        ret, img = cam.read()
        img = cv2.resize(img, (0,0), fx=scale, fy=scale) 
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        
        cluster_img = draw_clustered_flow(img, flow, counter)
        cv2.imshow('flow cluster', cluster_img)
        ## video.write(cluster_img)
        
        ## cv2.imshow('flow', draw_flow(gray, flow))
        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv2.imshow('glitch', cur_glitch)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print 'HSV flow visualization is', ['off', 'on'][show_hsv]
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print 'glitch is', ['off', 'on'][show_glitch]
        counter += 1.0
            
    cv2.destroyAllWindows()
    ## video.release()

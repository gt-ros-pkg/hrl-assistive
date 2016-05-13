import os, sys
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import svm
from matplotlib.patches import Rectangle   

MAT_WIDTH = 0.74#0.762 #metres
MAT_HEIGHT = 1.75 #1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 


class DetectHeadFromShoulders:
    '''Detects the head of a person sleeping on the autobed'''
    def __init__(self):
        self.feature_params = {'template_size': [24, 36], 'hog_cell_size': 6}
        self.train_path_pos = '/home/yashc/Desktop/dataset/pos_samples_head.p'
        self.train_path_neg = '/home/yashc/Desktop/dataset/neg_samples_head.p'

    def run_detector(self, test_scn_path, feature_params):
	test_scenes = pkl.load(open(test_scn_path, 'rb'))
	bboxes = np.zeros((0,4))
	confidences = np.zeros((0,1))
	num_scales = 1
	L = feature_params['template_size']
	step = np.array(feature_params['template_size']) / feature_params['hog_cell_size']
	hog_cell_size = feature_params['hog_cell_size']
	THRESH = 0.0
	count = 0
        for i in range(np.shape(test_scenes)[0]):
            print "Testing image {} of {}".format(i, np.shape(test_scenes)[0])
    	    img = test_scenes[i]
    	    img = img/100.
    	    temp_bboxes = np.zeros((0, 4))
    	    temp_confidences = np.zeros((0,1))
            curr_image = img
            for row in xrange(0,np.shape(curr_image)[0] - L[0], hog_cell_size):
                for col in xrange(0,np.shape(curr_image)[1] - L[1], hog_cell_size):
                    cropped_img = curr_image[row:row+L[0], col:col+L[1]]
                    HOG, viz = hog(cropped_img, 
                                   orientations=9, 
                                   pixels_per_cell=(6,6), 
                                   cells_per_block=(6,4), 
                                   visualise=True)
                    
                    test_score = self.clf.predict(HOG)
                    if test_score > THRESH:
                        bbox = np.zeros((1,4))
                        bbox[:, 0] = np.floor(col)
                        bbox[:, 1] = np.floor(row)
                        bbox[:, 2] = np.floor(np.floor(col) + L[1] -1)
                        bbox[:, 3] = np.floor(np.floor(row) + L[0] -1)
                        temp_bboxes = np.vstack((temp_bboxes, bbox))   
            try:
                bboxes = np.vstack((bboxes, temp_bboxes[1 , :])) 
            except:
                temp_bbox = np.array([0., 0., 23., 35.])
                bboxes = np.vstack((bboxes, temp_bbox))
        return bboxes[1:, :]

    def detect_blob(self, test_scn_path):
        '''Computes blobs in pressure map'''
	    test_scenes = pkl.load(open(test_scn_path, 'rb'))

        for i in range(np.shape(test_scenes)[0]):
            p_map = test_scenes[i]
            weights = np.zeros(np.shape(p_map))
            for i in range(np.shape(p_map)[0]):
                weights[i, :] = self.sigmoid((np.shape(p_map)[0]/8.533 - i))
            p_map = np.array(weights)*np.array(p_map)
            #plt.matshow(p_map)
            #plt.show()
            blobs = blob_doh(p_map, 
                         min_sigma=1, 
                         max_sigma=7, 
                         threshold=20,
                         overlap=0.1) 
        return blobs


    def visualize_detections_by_image(self, bboxes, test_scn_path, feature_params):
        '''Display head location'''
	test_scenes = pkl.load(open(test_scn_path, 'rb'))
        fig = plt.gcf()

        for i in range(np.shape(test_scenes)[0]):
            pressure_map_matrix = test_scenes[i]
            fig = plt.figure()
            ax1 = fig.add_subplot(111, aspect='equal')
            ax1.imshow(pressure_map_matrix, interpolation='nearest', cmap=
                plt.cm.bwr, origin='upper', vmin=0, vmax=100)
            xlim = [0.0, 54.0]
            ylim = [128.0, 0.0]                     
            plt.xlim(xlim)
            plt.ylim(ylim)                         
            ax1.add_patch(Rectangle((bboxes[i, 0], bboxes[i, 1]), 
                                    feature_params['template_size'][1], 
                                    feature_params['template_size'][0], 
                                    fill=False, 
                                    alpha=1,
                                    edgecolor="green",
                                    linewidth=2
                                    ))
            plt.show()

    def run(self):
        '''Detects head location using bounding box'''
        self.clf = pkl.load(open('./svm_classifier.p', 'rb'))
        test_scn_path = '/home/yashc/Desktop/dataset/whole_body_samples.p'
        print test_scn_path
        bboxes = self.run_detector(test_scn_path, self.feature_params)
        self.visualize_detections_by_image(bboxes, test_scn_path, self.feature_params)


if __name__ == '__main__':
    head_shoulder = HeadShoulderDetector()
    head_shoulder.run()



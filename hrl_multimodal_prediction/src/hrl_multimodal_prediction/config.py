# Configuration File for all Constants
ROSBAG_PATH = './bagfiles/'
ROSBAG_TEST_PATH = './bagfiles/testbag/'
ROSBAG_UNPACK_PATH = './bagfiles/unpacked/'
PROCESSED_DATA_PATH = './processed_data/'
WEIGHT_FILE = './weights/real_data.h5'

BAG2DATA_UNPACK = False
BAG2DATA_TESTDATA = False

# Audio Params
RATE = 44100
N_MEL = 128
N_FFT = 4096
HOP_LENGTH = N_FFT/4 
N_MFCC = 3

IMAGE_DIM = 3 #image dimension
MFCC_DIM = 3 #audio dimension
INPUT_DIM = MFCC_DIM + IMAGE_DIM #total dimension in LSTM
TIMESTEP_IN = 1
TIMESTEP_OUT = 10
N_NEURONS = TIMESTEP_OUT

BATCH_SIZE = 64
NUM_BATCH = 200 #Total #samples = Num_batch x Batch_size
NB_EPOCH = 500
PRED_BATCH_SIZE = 1

PLOT = True
DENSE = True #True if TimeDistributedDense layer is used

# In Subscriber/Prediction modules
P_MFCC_TIMESTEP = 5#3 # when data collected with 4096 and n_fft = 8192

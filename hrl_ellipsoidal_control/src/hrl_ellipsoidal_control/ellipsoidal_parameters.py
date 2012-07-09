import numpy as np

ELL_LOCAL_VEL = 0.0025
LONGITUDE_STEP = 0.12
LATITUDE_STEP = 0.096
HEIGHT_STEP = 0.0986
ell_trans_params = {
    'translate_up' : (-LATITUDE_STEP, 0, 0),   'translate_down' : (LATITUDE_STEP, 0, 0),
    'translate_right' : (0, -LONGITUDE_STEP, 0), 'translate_left' : (0, LONGITUDE_STEP, 0),
    'translate_in' : (0, 0, -HEIGHT_STEP),      'translate_out' : (0, 0, HEIGHT_STEP)}

ELL_ROT_VEL = 0.002
ROLL_STEP = np.pi/12
PITCH_STEP = np.pi/12
YAW_STEP = np.pi/12
ell_rot_params = {
    'rotate_x_pos' : (-ROLL_STEP, 0, 0), 'rotate_x_neg' : (ROLL_STEP, 0, 0),
    'rotate_y_pos' : (0, PITCH_STEP, 0), 'rotate_y_neg' : (0, -PITCH_STEP, 0),
    'rotate_z_pos' : (0, 0, -YAW_STEP),  'rotate_z_neg' : (0, 0, YAW_STEP)}


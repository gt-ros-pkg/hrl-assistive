
import numpy as np

DANGEROUS_CB_COOLDOWN = 5.0
CONTACT_CB_COOLDOWN = 0.5
TIMEOUT_CB_COOLDOWN = 20.0

APPROACH_VELOCITY = 0.0025
GLOBAL_VELOCITY = 0.0025
HEIGHT_STEP = 0.17
LATITUDE_STEP = 0.12
LOCAL_VELOCITY = 0.0025
LONGITUDE_STEP = 0.06
RETREAT_HEIGHT = 1.65
SAFETY_RETREAT_HEIGHT = 1.9
SAFETY_RETREAT_VELOCITY = 0.0150
SLOW_RETREAT_VELOCITY = 0.0200
SHAVE_HEIGHT = 0.8
TIMEOUT_TIME = 30.0
HEIGHT_CLOSER_ADJUST = 0.5
TRIM_RETREAT_LATITUDE = 1.9

LAT_BOUNDS = {'r' : (np.pi/8, 7*np.pi/8), 'l' : (np.pi/8, 7*np.pi/8)}
#LON_BOUNDS = {'r' : (-np.inf, np.inf), 'l' : (-np.inf, np.inf)}
LON_BOUNDS = {'r' : (-5*np.pi/8, np.pi/8), 'l' : (-np.pi/8, 5*np.pi/8)}
HEIGHT_BOUNDS = {'r' : (0.2, 3.5), 'l' : (0.2, 3.5)}

outcomes_spa = ['succeeded','preempted','aborted']

#class TransitionIDs:
#    GLOBAL_START      =  1
#    GLOBAL_PREEMPT    =  2
#    GLOBAL_STOP       =  2
#    LOCAL_START       =  3 #TODO FIX
#    LOCAL_PREEMPT     =  2
#    LOCAL_STOP        =  4
#    SHAVE_START       =  8
#    MOVE_COLLISION        =  9
#    ELL_RETREAT_GLOBAL    = 10
#    ELL_MOVE_GLOBAL       = 11
#    ELL_APPROACH_GLOBAL   = 12
#    ELL_RETREAT_SLOW      = 13
#    ELL_RETREAT_FAST      = 14
#    HOLDING               = 15

class Messages:
    ENABLE_CONTROLLER    = "Enabling ellipsoidal controller. Arm may twitch as it switches controllers."
    NO_PARAMS_LOADED     = "Cannot enable ellipsoidal controller. Must first register head."
    ARM_AWAY_FROM_HEAD   = "Cannot enable ellipsoidal controller. Tool must be setup near head."
    DISABLE_CONTROLLER   = "Disabling ellipsoidal controller. You must reenable to give more commands."
    DANGEROUS_FORCE      = "Dangerous force over %.1f N detected, retreating from face."
    TIMEOUT_RETREAT      = "Timeout from lack of contact over %.0f s, retreating from face."
    CONTACT_FORCE        = "Arm stopped due to sensing contact force over %.1f N."
    GLOBAL_START         = "Global ellipsoid move to pose %s running."
    GLOBAL_PREEMPT       = "Global ellipsoid move to pose %s preempted."
    GLOBAL_SUCCESS       = "Global ellipsoid move to pose %s successful."
    LOCAL_START          = "Local ellipsoid move %s running."
    ROT_RESET_START      = "Running reset rotation movement. The tool should rotate perpendicular to the head."
    LOCAL_PREEMPT        = "Local ellipsoid move %s preempted."
    LOCAL_SUCCESS        = "Local ellipsoid move %s successful."


button_names_dict = {
    'translate_up'       : "translate up",
    'translate_down'     : "translate down",
    'translate_left'     : "translate left",
    'translate_right'    : "translate right",
    'translate_in'       : "translate in",
    'translate_out'      : "translate out",

    'rotate_y_neg'       : "rotate down",
    'rotate_y_pos'       : "rotate up",
    'rotate_z_neg'       : "rotate left",
    'rotate_z_pos'       : "rotate right",
    'rotate_x_neg'       : "rotate clockwise",
    'rotate_x_pos'       : "rotate counter clockwise",

    'reset_rotation'     : "reset rotation",
}

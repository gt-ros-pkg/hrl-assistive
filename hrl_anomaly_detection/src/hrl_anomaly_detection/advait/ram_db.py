
import numpy as np, math
import os
db_path = os.environ['HRLBASEPATH']+'/src/projects/modeling_forces/handheld_hook'


## import roslib; roslib.load_manifest('modeling_forces')
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import hrl_lib.util as ut
import mechanism_analyse_RAM as mar


k = 'kitchen'
r = 'refrigerator'
f = 'freezer'
c = 'cabinet'
dr = 'drawer'
t = 'toolchest'
b = 'broiler'
s = 'springloaded'
do = 'door'
m = 'microwave'
be = 'bedroom'
o = 'office'
te = 'test set'
ig = 'ignore'
ro = 'robot'

tags_dict = {
'advait_freezer': [k, f],
'advait_oven_drawer': [k, dr, b],
'advait_refrigerator': [k, r],
'hai_freezer': [k, f],
'hai_kitchen_cabinet_lower_left': [k, c],
'hai_kitchen_cabinet_lower_right': [k, c],
'hai_kitchen_cabinet_upper_left': [k, c],
'hai_kitchen_cabinet_upper_right': [k, c],
'hai_kitchen_drawer_1': [k, dr],
'hai_refrigerator': [k, r],
'hai_rooom_drawer': [be, dr],
'HRL_lab_cabinet_recessed_right': [o, c],
'HRL_toolchest_drawer_empty': [dr, t],
'HRL_toolchest_drawer_filled': [dr, t],
'HSI_Executive_Board_Room_Cabinet_Right': [o, c],
'HSI_Executive_Board_Room_Cabinet_Left': [o, c],
#'HSI_Glass_Door': [do, s],
'HSI_kitchen_cabinet_left': [o, c],
'HSI_kitchen_cabinet_right_charlie': [o, c],
'HSI_Suite_210_Clear_Cabinet_Right': [o, c],
'HSI_Suite_210_brown_cabinet_right': [o, c],
'HSI_lab_cabinet_recessed_left': [o, c],
#'HSI_spring_loaded_glass_door2': [do, s, o],
'Jason_freezer': [k, f],
'Jason_neighbor_freezer1': [k, f],
'Jason_neighbor_kitchen_cabinet_long_right': [k, c],
'Jason_neighbor_refrigerator': [k, r],
'Jason_refrigerator': [k, r],
'Jason_TV_cabinet_right': [be, c],
'naveen_freezer': [k, f],
'naveen_kitchen_cabinet_left': [k, c],
'naveen_kitchen_drawer_top': [k, dr],
'naveen_microwave': [k, m],
'naveen_oven_drawer': [k, dr, b],
'naveen_refrigerator': [k, r],
'Neils_Kitchen_Cabinet_Left': [k, c],
'Neils_Kitchen_Drawer': [k, dr],
'Neils_Kitchen_Fridge': [k, r],
'Neils_Kitchen_Oven_Drawer': [k, dr, b],
'Neils_Kitchen_Sink_Cabinet_Right': [k, c],
'Neils_Room_Desk_Drawer_Bottom': [be, dr],
'Neils_Room_Desk_Drawer_middle': [be, dr],
'Neils_Room_Desk_Drawer_Top': [be, dr],
'richard_hex_side_cabinet': [be, c],
'richard_plastic_drawer': [be, dr],
#------ testing mechanisms ------------
'ikea_cabinet_cody': [o, c, te],
'ikea_cabinet_collision_cody': [o, c, te],
'ikea_cabinet_pr2': [o, c, te],
'lab_spring_loaded_cody': [do, s, te],
'lab_spring_collision_cody': [do, s, te],
'kitchen_cabinet_pr2': [o, c, te],
#-------- others, to be ignored -------
'ikea_cabinet_cody_10cm_cody': [o, c, ig],
'ikea_cabinet_cody_2.5cm_cody': [o, c, ig],
'ikea_cabinet_cody_5cm_cody': [o, c, ig],
'ikea_cabinet_thick_10cm_cody': [o, c, ig],
'ikea_cabinet_thick_2.5cm_cody': [o, c, ig],
'ikea_cabinet_thick_5cm_cody': [o, c, ig],

'ikea_cabinet_pos1_cody': [o, c, ig],

'HSI_kitchen_cabinet_right_advait': [o, c, ig],
'HSI_kitchen_cabinet_right_tiffany': [o, c, ig],

#'ikea_cabinet_pos1_10cm_cody': [o, c, ig],
#'ikea_cabinet_pos1_2.5cm_cody': [o, c, ig],
#'ikea_cabinet_pos1_5cm_cody': [o, c, ig],
#'ikea_cabinet_pos1_cody': [o, c, ig],
#'ikea_cabinet_pos2_10cm_cody': [o, c, ig],
#'ikea_cabinet_pos2_2.5cm_cody': [o, c, ig],
#
#'ikea_cabinet_move_pos1_5cm_cody': [o, c, ig],
#
#'kitchen_cabinet_dec6_cody': [o, c, ig],
#'kitchen_cabinet_dec7_10hz_separate_ft_pr2': [o, c, ig],
#
#'robot_ikea_cabinet_pos1_cody': [o, c],
#'robot_ikea_cabinet_pos1_pr2': [o, c],
#'robot_kitchen_cabinet_dec6_cody': [o, c],
#'robot_kitchen_cabinet_dec7_10hz_separate_ft_pr2': [o, c],
#'kitchen_cabinet_noisy_kinematics_cody': [o, c, ig],
#'kitchen_cabinet_noisy_kinematics_2_cody': [o, c, ],
#'kitchen_cabinet_noisy_kinematics_pr2': [o, c, ],
#'kitchen_cabinet_noisy_kinematics_uniform_cody': [o, c],
#'ikea_cabinet_noisy_kinematics_cody': [o, c, ig],


'kitchen_cabinet_cody': [o, c, ro],
'kitchen_cabinet_pr2': [o, c, ro],
'ikea_cabinet_cody': [o, c, ro],
'ikea_cabinet_pr2': [o, c, ro],
'lab_fridge_cody': [k, r, ro],
#'lab_spring_door_cody': [do, s],

'kitchen_cabinet_noisy_cody': [o, c, ro],
'kitchen_cabinet_noisy_pr2': [o, c, ro],
'ikea_cabinet_noisy_cody': [o, c, ro],
'ikea_cabinet_noisy_pr2': [o, c, ro],
'lab_fridge_noisy_cody': [k, r, ro],
'lab_spring_door_noisy_cody': [do, s, ig],

'kitchen_cabinet_known_rad_cody': [o, c, ro],
'kitchen_cabinet_known_rad_pr2': [o, c, ro],
'ikea_cabinet_known_rad_cody': [o, c, ro],
'ikea_cabinet_known_rad_pr2': [o, c, ro],
'lab_fridge_known_rad_cody': [k, r, ro],
'lab_spring_door_known_rad_cody': [do, s, ro],


}


def get_mechanism_names(has_tags=[], avoid_tags=[]):
    mech_nm_list = []
    for m in tags_dict.keys():
        is_mech = True
        for h in has_tags:
            if h not in tags_dict[m]:
                is_mech = False
        for a in avoid_tags:
            if a in tags_dict[m]:
                is_mech = False
        # if I want to get a specific mechanism by name
        if has_tags[0] == m:
            is_mech = True
        if is_mech:
            mech_nm_list.append(m)
    return mech_nm_list

def get_ref_dicts(has_tags=[], avoid_tags=[]):
    nm_list = get_mechanism_names(has_tags, avoid_tags)
    d_list = [ut.load_pickle(db_path+'/RAM_db/'+nm+'.pkl') for nm in nm_list]
    return d_list

def get_mean_std_config(has_tags=[], avoid_tags=[]):
    d_list = get_ref_dicts(has_tags, avoid_tags)
    # combined reference trajectory
    max_config_list = [np.max(d['config']) for d in d_list]
    lim_config = np.min(max_config_list)
    lim_idx = np.argmin(max_config_list)

    combined_config =None
    vll = []
    for i, d in enumerate(d_list):
        # truncate to min config
        idx = list(d['config']).index(lim_config)
        vl = [v[:idx+1] for v in d['vec_list']]
        vll.append(vl)
        if i == lim_idx:
            combined_config = d['config']

    vec_list_combined = flatten_list(vll)
    comb_mn, comb_std = mean_std_force_traj(vec_list_combined)
    return comb_mn, comb_std, combined_config, d['typ']

## convert list of lists into a list
def flatten_list(ll):
    return [i for j in ll for i in j]

def mean_std_force_traj(vec_list):
    vec_mat = np.matrix(vec_list)
    mean_force_traj = vec_mat.mean(0).A1
    std_force_traj = vec_mat.std(0).A1
    return mean_force_traj.tolist(), std_force_traj.tolist()

def query_and_plot(has_tags=[], avoid_tags=[]):
    d_list = get_ref_dicts(has_tags, avoid_tags)

    # plot individually
    for d in d_list:
        mar.plot_reference_trajectory(d['config'], d['mean'],
                                d['std'], d['typ'], d['name'])

    comb_mn, comb_std, combined_config = get_mean_std_config(has_tags,
                                                             avoid_tags)
    mar.plot_reference_trajectory(combined_config, comb_mn, comb_std,
                                  d['typ'], 'Combined Plot')



if __name__ == '__main__':
    import matplotlib.pyplot as pp

    l1 = [k]
    l2 = [c, b, dr]
    print get_mechanism_names(l1, l2)
    query_and_plot(l1, l2)
    pp.show()







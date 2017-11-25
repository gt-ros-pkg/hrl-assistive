#!/bin/bash

# Generate dae from URDF
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_body_v2/robots/bed_and_body_v2_low_res.URDF ./collada/bed_and_body_v2_low_res.dae 

# If joint axes/anchors have pretty big decmials, round off those (using moveit_ikfast's script).
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_body_v2_low_res.dae ./collada/bed_and_body_v2_low_res_rounded.dae 3

# Generate dae from URDF
rosrun collada_urdf urdf_to_collada ./urdf/autobed_simulation/robots/autobed_simulation_normal.URDF ./collada/autobed_simulation_normal.dae 
rosrun collada_urdf urdf_to_collada ./urdf/autobed_simulation/robots/autobed_simulation_expanded.URDF ./collada/autobed_simulation_expanded.dae 
rosrun collada_urdf urdf_to_collada ./urdf/wheelchair_simulation/robots/wheelchair_simulation_normal.URDF ./collada/wheelchair_simulation_normal.dae 
rosrun collada_urdf urdf_to_collada ./urdf/wheelchair_simulation/robots/wheelchair_simulation_expanded.URDF ./collada/wheelchair_simulation_expanded.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/wheelchair_and_body_assembly/robots/wheelchair_and_body_assembly.URDF ./collada/wheelchair_and_body_assembly.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_body_v3_real_expanded/robots/bed_and_body_v3_with_wall.URDF ./collada/bed_and_body_v3_real_expanded.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_body_expanded/robots/bed_and_body_expanded.URDF ./collada/bed_and_body_expanded.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_environment_mannequin/robots/bed_and_environment_mannequin_openrave.URDF ./collada/bed_and_environment_mannequin.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_environment_henry/robots/bed_and_environment_henry.URDF ./collada/bed_and_environment_henry.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_environment_henry/robots/bed_and_environment_lab.URDF ./collada/bed_and_environment_lab.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_environment_henry_tray/robots/bed_and_environment_henry_tray.URDF ./collada/bed_and_environment_henry_tray.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/wheelchair_henry/robots/wheelchair_henry.URDF ./collada/wheelchair_henry.dae 
#rosrun collada_urdf urdf_to_collada ./models/human.urdf ./collada/human.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_body_v3_real_expanded/robots/bed_and_body_v3.URDF ./collada/bed_and_body_v3_real_expanded_no_wall.dae 

# If joint axes/anchors have pretty big decmials, round off those (using moveit_ikfast's script).
rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/autobed_simulation_normal.dae ./collada/autobed_simulation_normal_rounded.dae 3
rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/autobed_simulation_expanded.dae ./collada/autobed_simulation_expanded_rounded.dae 3
rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/wheelchair_simulation_normal.dae ./collada/wheelchair_simulation_normal_rounded.dae 3
rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/wheelchair_simulation_expanded.dae ./collada/wheelchair_simulation_expanded_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/wheelchair_and_body_assembly.dae ./collada/wheelchair_and_body_assembly_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_body_v3_real_expanded.dae ./collada/bed_and_body_v3_real_expanded_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_environment_mannequin.dae ./collada/bed_and_environment_mannequin_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_environment_henry.dae ./collada/bed_and_environment_henry_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_environment_lab.dae ./collada/bed_and_environment_lab_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_body_expanded.dae ./collada/bed_and_body_expanded_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_environment_henry_tray.dae ./collada/bed_and_environment_henry_tray_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/wheelchair_henry.dae ./collada/wheelchair_henry_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/human.dae ./collada/human_rounded.dae 4
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_body_v3_real_expanded_no_wall.dae ./collada/bed_and_body_v3_real_expanded_no_wall_rounded.dae 3


# Generate dae from URDF
#rosrun collada_urdf urdf_to_collada ./urdf/wheelchair_best/robots/wheelchair_best.URDF ./collada/wheelchair.dae 

# If joint axes/anchors have pretty big decmials, round off those (using moveit_ikfast's script).
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/wheelchair.dae ./collada/wheelchair_rounded.dae 3

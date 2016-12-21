#!/bin/bash

# Generate dae from URDF
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_body_v2/robots/bed_and_body_v2_low_res.URDF ./collada/bed_and_body_v2_low_res.dae 

# If joint axes/anchors have pretty big decmials, round off those (using moveit_ikfast's script).
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_body_v2_low_res.dae ./collada/bed_and_body_v2_low_res_rounded.dae 3

# Generate dae from URDF
#rosrun collada_urdf urdf_to_collada ./urdf/wheelchair_and_body_assembly/robots/wheelchair_and_body_assembly.URDF ./collada/wheelchair_and_body_assembly.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_body_expanded/robots/bed_and_body_expanded.URDF ./collada/bed_and_body_expanded.dae 
rosrun collada_urdf urdf_to_collada ./urdf/bed_and_environment_henry_tray/robots/bed_and_environment_henry_tray.URDF ./collada/bed_and_environment_henry_tray.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/wheelchair_henry/robots/wheelchair_henry.URDF ./collada/wheelchair_henry.dae 
#rosrun collada_urdf urdf_to_collada ./models/human.urdf ./collada/human.dae 
#rosrun collada_urdf urdf_to_collada ./urdf/bed_and_body_v3_nowall/robots/bed_and_body_v3_nowall.URDF ./collada/bed_and_body_v3_nowall.dae 

# If joint axes/anchors have pretty big decmials, round off those (using moveit_ikfast's script).
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/wheelchair_and_body_assembly.dae ./collada/wheelchair_and_body_assembly.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_body_expanded.dae ./collada/bed_and_body_expanded_rounded.dae 3
rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_environment_henry_tray.dae ./collada/bed_and_environment_henry_tray_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/wheelchair_henry.dae ./collada/wheelchair_henry_rounded.dae 3
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/human.dae ./collada/human_rounded.dae 4
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/bed_and_body_v3_nowall.dae ./collada/bed_and_body_v3_nowall_rounded.dae 3


# Generate dae from URDF
#rosrun collada_urdf urdf_to_collada ./urdf/wheelchair_best/robots/wheelchair_best.URDF ./collada/wheelchair.dae 

# If joint axes/anchors have pretty big decmials, round off those (using moveit_ikfast's script).
#rosrun hrl_gazebo_darci round_collada_numbers.py ./collada/wheelchair.dae ./collada/wheelchair_rounded.dae 3

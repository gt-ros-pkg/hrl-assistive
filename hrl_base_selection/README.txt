Instructions on how to run base selection:



Connect to PR2
Open screen with multiple windows
rosrun pr2_dashboard pr2_dashboard
roslaunch assistive_teleop assistive_teleop.launch
roslaunch hrl_face_adls face_adls.launch subject:=dummy_1
roslaunch hrl_haptic_mpc start_pr2_mpc.launch arm:=l
roslaunch hrl_fabric_based_tactile_sensor pr2_tactile_sleeve_combined.launch
rosrun hrl_base_selection base_selection_service_pkl.py
roslaunch hrl_base_selection arm_reacher.launch 
roslaunch RYDS start_pr2_mpc_l.launch

Then go to 
http://monty1.hsi.gatech.edu:8000/assistive_teleop/hz_tab_interface.html#skin
and use the web interface to control the robot.

Note that this needs
svn/robot1/usr/kelsey/rfh_data
which is not necessarily something obvious to need.

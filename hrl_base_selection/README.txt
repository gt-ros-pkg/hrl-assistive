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


To get this working / wake up from hibernation:
Run the autobed from mac-mini-marvin or another computer with hrl_autobed_dev
roslaunch /hrl_autobed_dev/autobed_web/scripts/autobed_start_all_marvin.sh

Make sure to change the PR2 arm gains in the package: /hrl_haptic_manip_dev/hrl_haptic_mpc/scripts/change_gains.sh
Set rosmaster pointed at the PR2.

DO NOT PLUG INTERNET FROM BUILDING INTO THE SERVICE PORT. But connect the computer to the PR2 using the service port to be able to visualize the kinect. However, visualize pressure mat data from the mac-mini to the internet to the PR2 to the computer you are running rviz on. In order to communicate through the service port you have to type in sudo ifconfig eth0:0 10.68.0.20 (make sure you have a local area connection). This makes an alias or fake connection of eth0. 

PR2 has 3 computers: C1 is 10.68.0.1 and C2 is 10.68.0.2. head is something else that's not .20. By default you'll connect to C1 on the pr2 upon ssh. FTsensor is 192.168.0.124 (which is FT8)
e.g. sudo ifconfig wan0:0 192.168.0.124. previously you couldn't address it but NOW you can using ping 192.168.0.124.

Run the hrl_base_selection under henry_pose_devel on the PR2. ssh into the PR2 through a 
roslaunch /hrl_base_selection/launch/pose_est_henryc_objects.launch
roslaunch /hrl_base_selection/launch/base_selection_new_ui.launch





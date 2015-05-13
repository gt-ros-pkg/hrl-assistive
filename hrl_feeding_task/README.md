Steps to running all code for hrl_feeding_task (launch Kinect files, arm files, start robot, etc)

Start robot:

1) ssh dpark@pr2c1
2) rosrun pr2_dashboard pr2_dashboard 
3) roslaunch pr2_teleop teleop_joystick.launch

Launch Kinect files:

1) roslaunch hrl_feeding_task head_registration_Kinect2.launch
2) roslaunch hrl_feeding_task assistive_teleop_Kinect2.launch

Calibrate arms:

1) roscd hrl_haptic_mpc
2) ./change_gains_pr2.sh

To launch each arm:

1) roslaunch hrl_feeding_task start_pr2_mpc_l.launch
   roslaunch hrl_feeding_task start_pr2_mpc_r.launch
2) roslaunch hrl_feeding_task arm_reacher_l.launch
   roslaunch hrl_feeding_task arm_reacher_r.launch
3) rosrun hrl_feeding_task arm_reacher_client.py
	*) If you want to manually set the bowl location...
	roslaunch hrl_feeding_task bowl_location_publisher.py
4) rosrun hrl_feeding_task emergency_stop_publisher.py

To find end wrist location:

rosrun tf tf_echo /torso_lift_link /l_gripper_spoon_frame
rosrun tf tf_echo /torso_lift_link /r_gripper_tool_frame

Steps to running all code for hrl_feeding_task (launch Kinect files, arm files, start robot, etc)

Start robot:

	1) ssh dpark@pr2c1
	2) rosrun pr2_dashboard pr2_dashboard 
	3) roslaunch pr2_teleop teleop_joystick.launch

Launch Kinect files:

	1) roslaunch hrl_feeding_task head_registration_Kinect2.launch
	2) roslaunch hrl_feeding_task assistive_teleop_Kinect2.launch
	^DON'T WORK ANYMORE, INSTEAD DO...
	*) roslaunch hrl_feeding_task Feeding_Visual_Kinect2.launch

Calibrate arms:

	1) roscd hrl_feeding_task/launch
	2) ./change_gains_pr2.sh

To launch each arm:

	First, press Start on runstop

	1) Left arm:  roslaunch hrl_feeding_task start_pr2_mpc_l.launch
   	   Right arm: roslaunch hrl_feeding_task start_pr2_mpc_r.launch
		*) While these are running, the last position of the robot arm is 'saved' so if you stop and then start the runstop, it will return to last position. Be careful, make sure you shut off these nodes before restarting the robot with runstop
		
	2) Left arm:  roslaunch hrl_feeding_task arm_reacher_l.launch
   	   Right arm: roslaunch hrl_feeding_task arm_reacher_r.launch
   	   
	3) rosrun hrl_feeding_task arm_reacher_client.py

If you want to manually set the bowl location:

	1) roslaunch hrl_feeding_task bowl_location_publisher.py
	
Start emergency stop node, enter '!' to stop and reverse robot arm:

	1) rosrun hrl_feeding_task emergency_stop_publisher.py

Register bowl position:

	1) Go to http://pr2c1.hsi.gatech.edu:8000/assistive_teleop/hz_tab_interface.html
	2) Press RYDS tab, register bowl location

To manually set end effector position quickly and change between angles, used for testing:

	1) rosrun hrl_feeding_task arm_movement_tests.py

To find end wrist location:

	1) Left arm:  rosrun tf tf_echo /torso_lift_link /l_gripper_spoon_frame
	2) Right arm: rosrun tf tf_echo /torso_lift_link /r_gripper_tool_frame



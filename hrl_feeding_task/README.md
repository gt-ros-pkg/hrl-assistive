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

	1) Both arms: roslaunch hrl_feeding_task start_pr2_mpc_all.launch
		*) While these are running, the last position of the robot arm is 'saved' so if you stop and then start the runstop, it will quickly return to last position, sometimes dangerously fast.
		
	2) Both arms:  roslaunch hrl_feeding_task arm_reacher_all.launch
		
		This will launch two instances of mpcBaseAction (normal and 'right'), as well as arm_reacher_server.py and arm_reacher_helper_right.py. Each operates with respective right/left nodes & topics. 
   	   
	3) rosrun hrl_feeding_task arm_reacher_client.py
	
Start emergency stop node, enter '!' to stop and reverse robot arm:

	1) rosrun hrl_feeding_task emergency_stop_publisher.py

Register bowl and head position:

	1) Go to http://pr2c1.hsi.gatech.edu:8000/assistive_teleop/hz_tab_interface.html
	2) Press RYDS tab, register bowl location
	3) Go to Body Registration tab, register head location
	
Begin feeding task:

	1) In terminal window for arm_reacher_server.py node, follow on-screen instructions...
	2) Make sure bowl and head position are registered
	3) Initialize left and right arms
	4) Press 'y' to begin
	5) User is prompted for each step, press Enter to continue
	
Extra:
	
*To manually set the bowl location (in case Kinect location is not wanted):

	1) roslaunch hrl_feeding_task bowl_location_publisher.py

*To manually set end effector position quickly and change between angles, used for testing:

	1) roslaunch hrl_feeding_task arm_movement_tests.launch
	2) rosrun hrl_feeding_task arm_movement_tests.py

*To find end wrist location:

	1) Left arm:  rosrun tf tf_echo /torso_lift_link /l_gripper_spoon_frame
	2) Right arm: rosrun tf tf_echo /torso_lift_link /r_gripper_tool_frame

*To find head position after registering:

	1) rosrun tf tf_echo /torso_lift_link /head_frame






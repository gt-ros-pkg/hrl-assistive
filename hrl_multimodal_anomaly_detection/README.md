Healthcare Robotics Lab
=======================

Summer 2015 Project
-------------------

Multimodal Anomaly Detection
----------------------------

    Daehyung Park - Research Mentor
    Hyder Hasnain - Undergraduate
    Zackory Erickson - Undergraduate


...


Performing anomaly detection using multiple sources, including sound, audio, and vision data


Steps to run feeding code on PR2 and computer:
----------------------------------------------


Start robot:

    1) Set proper environment variables via .bashrc
       If using epipac via hhasnain3 user, run command "pr2"
       If using laptop via primary user, run command "pr2c1"
    2) ssh dpark@pr2c1
    3) rosrun pr2_dashboard pr2_dashboard 
    4) roslaunch pr2_teleop teleop_joystick.launch

Launch Kinect files for bowl and head registration:
    
    1) roslaunch hrl_feeding_task Feeding_Visual_Kinect2.launch

Launch Kinect files for vision tracking:

    1) 

Calibrate arms:

    1) roscd hrl_multimodal_anomaly_detection/launch/arm_control
    2) ./change_gains_pr2.sh

Launch arm code:

    First, press Start on runstop

    1) Both arms: roslaunch hrl_multimodal_anomaly_detection start_pr2_mpc_all.launch
        *) Launches haptic_mpc controller and associated topics/groups
        *) While these are running, the last position of the robot arm is 'saved' so if you stop and then start the runstop, it will quickly return to last position, sometimes dangerously fast.
        
    2) Both arms: roslaunch hrl_multimodal_anomaly_detection arm_reacher_all.launch
        *) Launches specific arm code and arm_reacher_feeding (specific arm service now)
        *) This will launch two instances of mpcBaseAction (normal and 'right'), as well as arm_reacher_server.py and arm_reacher_helper_right.py. Each operates with respective right/left nodes & topics. 

Launch data recording:

    1) rosrun hrl_multimodal_anomaly_detection local_data_record.py
        *) How it works...
            *) Launches local data recording node, reads recording settings and inputs
            *) Launches record_data stuff, which in turn enables the arm_reach_enable client
                ... and enables/starts the arm_reacher_feeding server
            *) At any time during running the feeding code, the data recording can be independently
                ... stopped, and if the local_data_record node is restarted while the arm_reacher_feeding
                ... server keeps running, the arm_reach_enable client will be re-activated and will 
                ... restart the arm_reacher_feeding service

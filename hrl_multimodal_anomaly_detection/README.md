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


Start robot (new terminal window, ssh'd into **robot**):

    1) Set proper environment variables via .bashrc
       If using epipac via hhasnain3 user, run command "pr2"
       If using laptop via primary user, run command "pr2c1"
    2) ssh dpark@pr2c1
    3) robot claim
    4) robot start
    5) rosrun pr2_dashboard pr2_dashboard 

*OPTIONAL*: Launch joystick teleop (new terminal window, ssh'd into **robot**):

    1) ssh dpark@pr2c1
    2) roslaunch pr2_teleop teleop_joystick.launch

Change arm gains and start mpc (new terminal window, ssh'd into **robot**):
    
    1) ssh dpark@pr2c1
    2) roscd hrl_multimodal_anomaly_detection/launch/arm_control
    3) ./change_gains_pr2.sh
    4) PRESS START ON RUNSTOP
    5) roslaunch hrl_multimodal_anomaly_detection start_pr2_mpc_all.launch


*OPTIONAL*: Launch Kinect files for bowl and head registration (new terminal window, ssh'd into **robot**):
    
    1) ssh dpark@pr2c1
    2) roslaunch hrl_feeding_task Feeding_Visual_Kinect2.launch

Launch arm feeding server (new terminal window, ssh'd into **robot**):
   
    1) ssh dpark@pr2c1
    2) Both arms: roslaunch hrl_multimodal_anomaly_detection arm_reacher_all.launch
    
Launch combined FT node, bowl publisher and data recording (new terminal window, running on **laptop**):
    
    1) roslaunch hrl_multimodal_anomaly_detection record_feeding_full.py
        
Launch data recording and **RUN** (new terminal window, running on **laptop**):

    1) rosrun hrl_multimodal_anomaly_detection local_data_record.py

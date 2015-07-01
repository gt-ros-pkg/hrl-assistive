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


Start robot (first terminal window, ssh'd into **robot**):

    1) Set proper environment variables via .bashrc
       If using epipac via hhasnain3 user, run command "pr2"
       If using laptop via primary user, run command "pr2c1"
    2) **ssh dpark@pr2c1**
    3) robot claim
    4) robot start
    5) rosrun pr2_dashboard pr2_dashboard 

Launch joystick teleop (second terminal window, ssh'd into **robot**):

    1) roslaunch pr2_teleop teleop_joystick.launch

Change arm gains and start mpc (third terminal window, ssh'd into **robot**):
    
    1) roscd hrl_multimodal_anomaly_detection/launch/arm_control
    2) ./change_gains_pr2.sh
    3) PRESS START ON RUNSTOP
    4) roslaunch hrl_multimodal_anomaly_detection start_pr2_mpc_all.launch


*OPTIONAL*: Launch Kinect files for bowl and head registration:
    
    1) roslaunch hrl_feeding_task Feeding_Visual_Kinect2.launch

Launch FT node (fourth terminal window, running on **laptop**):

    1) rosrun netft_rdt_driver netft_node 10.68.0.120 --rate 10

Launch bowl publisher (fifth terminal window, running on **laptop**):

    1) rosrun hrl_multimodal_anomaly_detection manual_bowl_head_pose_publisher.py

Launch arm feeding server (sixth terminal window, ssh'd into **robot**):
   
    1) Both arms: roslaunch hrl_multimodal_anomaly_detection arm_reacher_all.launch
        
Launch data recording and **RUN** (seventh terminal window, running on **laptop**):

    1) rosrun hrl_multimodal_anomaly_detection local_data_record.py
    2) *Follow prompts on terminal screen*

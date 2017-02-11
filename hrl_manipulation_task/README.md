Healthcare Robotics Lab
=======================

-------------------

Manipulation 
----------------------------

    Daehyung Park - Research Mentor
    Youkeun Kim - Undergraduate
    Zackory Erickson - Graduate Student

...


Performing anomaly detection using multiple sources, including sound, audio, and vision data


Steps to run feeding code on PR2 and computer:
----------------------------------------------


Start robot (**new** terminal window, ssh'd into **robot**):

    1) Set proper environment variables via .bashrc
       If using epipac via hhasnain3 user, run command "pr2"
       If using laptop via primary user, run command "pr2c1"
    2) ssh dpark@pr2c1
    3) robot claim
    4) robot start
    5) rosrun pr2_dashboard pr2_dashboard 


Change arm gains and start mpc (**new** terminal window, ssh'd into **robot**):
    
    1) ssh dpark@pr2c1
    2) roscd hrl_manipulation_task/launch/
    3) ./change_gains_pr2.sh
    4) PRESS START ON RUNSTOP
    5) roslaunch hrl_manipulation_task start_pr2_mpc_all.launch

To enable kinect & audio sensors:

    1) Press runstop on
    2) Turn on Kinect V2 Backpack
    3) ssh administrator@10.68.0.173
       Check if the IP address is correct using rostopic info 'camera topic'
    4) cd ~/hark/hrl-assistive/hrl_sound_localization/hark
    5) screen
    5.1) ./1_Julius.sh
    5.2) ./online_recog_colcheck.n

To start data recording
    1) roslaunch hrl_manipulation_task start_pr2_mpc_all.launch
    2) roslaunch hrl_manipulation_task sensor.launch
       Please check F/T sensor, Kinect, Audio device before running this launch file.
    3) roslaunch hrl_manipulation_task arm_reacher_all.launch
    4) (Option) xxxx.py for visualization
    5.1) (only data recording) rosrun hrl_manipulation_task arm_reacher_logging.py
    5.1) (with anomaly detection) rosrun hrl_manipulation_task arm_reacher_logging.py --dp
       

To calibrate vision offest (for mouth detection)
    1) rosrun hrl_manipulation_task findMouth.py --renew    








Launch arm feeding server (**new** terminal window, ssh'd into **robot**):
   
    1) ssh dpark@pr2c1
    2) Both arms: roslaunch hrl_multimodal_anomaly_detection arm_reacher_all.launch
    
        
Installation:
----------------------------
1) python_speech_features: https://github.com/jameslyons/python_speech_features     
2) sudo apt-get install ros-indigo-sound-play ros-indigo-ar-track-alvar ros-indigo-ar-track-alvar-msgs


GUI installation
----------------------------
    1) Installation sudo apt-get install ros-indigo-pr2-common-action-msgs ros-indigo-rosbridge-server ros-indigo-mjpeg-server ros-indigo-web-video-server
    2) Install jqueryui
        roscd assistive_teleop
        wget https://jqueryui.com/resources/download/jquery-ui-1.11.4.zip
        unzip jquery-ui-1.11.4.zip
        cd www && ln -s ../jquery-ui-1.11.4 jqueryui
    3) Install hrl-pr2-behaviors, hrl-sensor-utils

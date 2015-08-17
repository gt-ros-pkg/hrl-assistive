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


Start robot (**new** terminal window, ssh'd into **robot**):

    1) Set proper environment variables via .bashrc
       If using epipac via hhasnain3 user, run command "pr2"
       If using laptop via primary user, run command "pr2c1"
    2) ssh dpark@pr2c1
    3) robot claim
    4) robot start
    5) rosrun pr2_dashboard pr2_dashboard 

*OPTIONAL*: Launch joystick teleop (**new** terminal window, ssh'd into **robot**):

    1) ssh dpark@pr2c1
    2) roslaunch pr2_teleop teleop_joystick.launch

Record proper bowl position manually:

	*NOTE* Do this *before* changing custom gains since by default, the arms are stiffer and will provide a more accurate end effector position
	(Arm will sag after letting go, otherwise have someone else hold the arm/gripper in the proper place while recording tf spoon position)

	1) Make sure runstop is on **STOP**
	2) Position end of spoon to align with yellow markers on sides of bowl and bottom lip of bowl
	3) Press **START** on runstop

	On the **laptop** run
	4) rosrun tf tf_echo /torso_lift_link /l_gripper_spoon_frame
	
	5) Record position of "l_gripper_spoon_frame" ie: 

		At time 1435778466.052
			- Translation: [**0.923,** **0.286,** **-0.325**]
			- Rotation: in Quaternion [0.631, 0.324, -0.367, 0.602]
            	in RPY [1.582, 1.023, -0.064] 

    On the **laptop** (because we run this file from the laptop, not the robot)
    6) Enter this position into line 37-39 of manual_bowl_head_pose_publisher.py ie:

    	37| (self.bowl_pose_manual.pose.position.x, 
		38|	self.bowl_pose_manual.pose.position.y, 
		39|	self.bowl_pose_manual.pose.position.z) = (**0.923,** **0.286,** **-0.325**)

	7) If necessary, perform similiar steps for setting the head/mouth position

Change arm gains and start mpc (**new** terminal window, ssh'd into **robot**):
    
    1) ssh dpark@pr2c1
    2) roscd hrl_feeding_task/launch/
    3) ./change_gains_pr2.sh
    4) PRESS START ON RUNSTOP
    5) roslaunch hrl_multimodal_anomaly_detection start_pr2_mpc_all.launch

To start netft topic:

    1) rosrun netft_rdt_driver netft_node 10.68.0.120 --rate 10


*OPTIONAL*: Launch Kinect files for bowl and head registration (**new** terminal window, ssh'd into **robot**):
    
    1) ssh dpark@pr2c1
    2) roslaunch hrl_feeding_task Feeding_Visual_Kinect2.launch

Launch arm feeding server (**new** terminal window, ssh'd into **robot**):
   
    1) ssh dpark@pr2c1
    2) Both arms: roslaunch hrl_multimodal_anomaly_detection arm_reacher_all.launch
    
Launch combined FT node, bowl publisher and data recording (**new** terminal window, running on **laptop**):
    
    1) roslaunch hrl_multimodal_anomaly_detection record_feeding_full.launch
        

---

Both data recording and online anomaly detection can be done through `record_feeding_full.py`.

After recording a set of feeding/scooping observation sequences, we can begin to setup training for an HMM.
To begin, we must edit the data file locations within `dataFiles()` in `hmm/launcher_4d.py`.
For instance, we need to change the following lines:
```
    fileNamesTrain = ['/path_to_training_data/iteration_%d_success.pkl']
    iterationSetsTrain = [xrange(6)]
    fileNamesTest = ['/path_to_test_data/iteration_%d_success.pkl', '/path_to_test_data_2/iteration_%d_failure.pkl']
    iterationSetsTest = [xrange(4), xrange(3)]
```

Here fileNames are a list of file locations that contain `%d` to represent the iteration number of each subsequent observation.
The specific number of iterations to load can be specified as a list of integer lists, iterationSets.
For example, `iterationSetsTest = [xrange(4), xrange(3)]` can be used to load 4 successful data observations from one file location, and 3 failure data observations from another file location specified in `fileNamesTest`.

Once the training and test data locations have been specified, we can train a new HMM by using:
```
    rosrun hrl_multimodal_anomaly_detection launcher_4d.py
```

If an HMM has already been trained, you can train a new HMM by deleting previously trained models in `hmm/models` or by changing the HMM model location in the `trainMultiHMM()` function of `launcher_4d.py`:
```
    hmm = learning_hmm_multi_4d(nState=20, nEmissionDim=4)
    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, xData4=pdfSample, ***Change This --> ml_pkl='modals/ml_4d_bowl3.pkl'***, use_pkl=True)
```

Online anomaly detection can be updated using similar methods.
First we must retrain the online HMM model. This can be done by changing the training data locations within `setupMultiHMM()` of `onlineHMMLauncher.py`; similar to path locations described above.

Online anomaly detection can then be performed using the command:
```
    roslaunch hrl_multimodal_anomaly_detection record_feeding_full.py
```

Both left and right arm use setPostureGoal() properly, not repeated messages/commands sent to /*/haptic_mpc/goal_posture

Both left and right arm use setOrientGoal() IMPROPERLY, repeated messages/commands sent to /*/haptic_mpc/goal_pose

Usually 3-4 seconds between sent messages...
First received message is correct position and orientation. 
After a few seconds arm stops moving and goal_pose displays incorrect position and orientation received, and real-life position and orientation matches this incorrect message

Some ideas:
Two messages are sent/received to/from /*/haptic_mpc/goal_pose
	1) arm_reacher_server calls setOrientGoal(pos, quat, timeout) and PUBLISHES to /*/haptic_mpc/goal_pose
	   /*/haptic_mpc/goal_pose receives initial GOAL position to REACH 
	2) setOrientGoal(pos, quat, timeout) function goes through this process:
		setOrientGoal()
		 self.checkMovement()
		  self.checkInGoal()
		   self.setStop()
		    self.setStopPos()
		     **self.goal_pos_pub.publish(ps)**
			^ Publishes AGAIN to /goal_pose once movement is finished...
			  Movement is finished AFTER length of "timeout" value
	
--------------------------------------------
Demonstration of right arm procedure Test 1:
--------------------------------------------

1) setPostureGoal([-1.570, 0, 0, -1.570, 3.141, 0, -4.712], 7)
	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433357499.055512] Updating MPC weights. Pos: 0.0, Orient: 0.0, Posture: 5.0
		[INFO] [WallTime: 1433357507.001473] Got new pose trajectory
		[WARN] [WallTime: 1433357507.001935] Received empty pose array. Clearing trajectory buffer
	B) rostopic echo /right/haptic_mpc/goal_posture displays:
		header: 
		seq: 1
		stamp: 
		secs: 0
		nsecs: 0
		frame_id: ''
		data: [-1.57, 0.0, 0.0, -1.57, 3.141, 0.0, -4.712]
		---
2) setOrientGoalRight([.5, -.5, 0], [0.612372435696, 0.353553390593, -0.353553390593, 0.612372435696], 10)
	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433357830.077301] Got new goal pose
		[INFO] [WallTime: 1433357830.077316] Updating MPC weights. Pos: 5.0, Orient: 5.0, Posture: 0.0
		[INFO] [WallTime: 1433357841.000984] Got new pose trajectory
10 sec >	[INFO] [WallTime: 1433357841.001339] Got new goal pose
		[WARN] [WallTime: 1433357841.001785] Received empty pose array. Clearing trajectory buffer
		[INFO] [WallTime: 1433357841.040704] MPC entered deadzone: pos 0.000243202041771 (0.01); orient 0.0317483061436 (10.0)

	B) rostopic echo /right/haptic_mpc/goal_pose displays:
		header: 
		  seq: 1
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
correct >	  position: 
		    x: 0.5
		    y: -0.5
		    z: 0.0
correct >	  orientation: 
		    x: 0.612372435696
		    y: 0.353553390593
		    z: -0.353553390593
		    w: 0.612372435696
		---
		header: 
		  seq: 2
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
incorrect >	  position: 
		    x: 0.535461637665
		    y: -0.467707608387
		    z: 0.00458886778293
incorrect >	  orientation: 
		    x: 0.672241583796
		    y: 0.0579463480864
		    z: -0.284722940045
		    w: 0.680930481894
		---

	********APPROX 10 SECOND DELAY BETWEEN MESSAGES********
	*******COINCIDENTALLY EQUAL TO SEND TIMEOUT VALUE******


3) Next iteration setOrientGoalRight([.5, -.5, 0], [0.683012701892, 0.183012701892, -0.183012701892, 0.683012701892], 10)
	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433358020.869145] Updating MPC weights. Pos: 5.0, Orient: 5.0, Posture: 0.0
		[INFO] [WallTime: 1433358020.869226] Got new goal pose
		[INFO] [WallTime: 1433358031.001860] Got new pose trajectory
10 sec >	[INFO] [WallTime: 1433358031.002102] Got new goal pose
		[WARN] [WallTime: 1433358031.002459] Received empty pose array. Clearing trajectory buffer
		[INFO] [WallTime: 1433358031.040972] MPC entered deadzone: pos 1.35973995551e-16 (0.01); orient 0.0 (10.0)

	B) rostopic echo /right/haptic_mpc/goal_pose displays:
		header: 
		  seq: 3
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
correct >	  position: 
		    x: 0.5
		    y: -0.5
		    z: 0.0
correct >	  orientation: 
		    x: 0.683012701892
		    y: 0.183012701892
		    z: -0.183012701892
		    w: 0.683012701892
		---
		header: 
		  seq: 4
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
incorrect >	  position: 
		    x: 0.565938113511
		    y: -0.477785662761
		    z: 0.0266537250765
incorrect >	  orientation: 
		    x: 0.685321624026
		    y: 0.105803301154
		    z: -0.215358632638
		    w: 0.687575881234
		---

4) WHEN CLOSING/EXITING ARM_REACHER_SERVER NODE RIGHT AFTER SENDING A setOrientGoalRight() COMMAND!
	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433358228.002622] Got new goal pose
		[INFO] [WallTime: 1433358228.002696] Updating MPC weights. Pos: 5.0, Orient: 5.0, Posture: 0.0
		[INFO] [WallTime: 1433358244.760912] MPC entered deadzone: pos 0.00999589560129 (0.01); orient 0.556208248408 (10.0)
	B) rostopic echo /right/haptic_mpc/goal_pose displays:
		header: 
		  seq: 5
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
correct >	  position: 
		    x: 0.5
		    y: -0.5
		    z: 0.0
correct >	  orientation: 
		    x: 0.707106781187
		    y: 0.0
		    z: 0.0
		    w: 0.707106781187
		---
	**After this the final position/orientation of the end effector was CORRECT**

--------------------------------------------
Demonstration of right arm procedure Test 2:
--------------------------------------------
1) setPostureGoal([-1.570, 0, 0, -1.570, 3.141, 0, -4.712], 7)
	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433358814.920742] MPC entered deadzone: pos 0.0099676359642 (0.01); orient 0.80764795065 (10.0)
		[INFO] [WallTime: 1433358818.496479] Updating MPC weights. Pos: 0.0, Orient: 0.0, Posture: 5.0
		[INFO] [WallTime: 1433358826.002259] Got new pose trajectory
		[WARN] [WallTime: 1433358826.003014] Received empty pose array. Clearing trajectory buffer
	B) rostopic echo /right/haptic_mpc/goal_posture displays:
		header: 
		  seq: 1
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: ''
		data: [-1.57, 0.0, 0.0, -1.57, 3.141, 0.0, -4.712]
		---
2) setOrientGoalRight([.5, -.5, 0], [0.612372435696, 0.353553390593, -0.353553390593, 0.612372435696], 10)
	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433359026.518553] Updating MPC weights. Pos: 5.0, Orient: 5.0, Posture: 0.0
		[INFO] [WallTime: 1433359026.518648] Got new goal pose
		[INFO] [WallTime: 1433359037.001848] Got new pose trajectory
		[WARN] [WallTime: 1433359037.002309] Received empty pose array. Clearing trajectory buffer
10 sec >	[INFO] [WallTime: 1433359037.002928] Got new goal pose
		[INFO] [WallTime: 1433359037.040979] MPC entered deadzone: pos 0.000113909854919 (0.01); orient 0.114057582507 (10.0)


	B) rostopic echo /right/haptic_mpc/goal_pose displays:
		header: 
		  seq: 1
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
		  position: 
		    x: 0.5
		    y: -0.5
		    z: 0.0
		  orientation: 
		    x: 0.612372435696
		    y: 0.353553390593
		    z: -0.353553390593
		    w: 0.612372435696
		---
		header: 
		  seq: 2
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
		  position: 
		    x: 0.538997777039
		    y: -0.462854529227
		    z: 0.00804457152026
		  orientation: 
		    x: 0.674689809913
		    y: 0.0502361114578
		    z: -0.295292316808
		    w: 0.67459057297
		---


3) Next iteration setOrientGoalRight([.5, -.5, 0], [0.683012701892, 0.183012701892, -0.183012701892, 0.683012701892], 5)
	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433359089.620846] Updating MPC weights. Pos: 5.0, Orient: 5.0, Posture: 0.0
		[INFO] [WallTime: 1433359089.620862] Got new goal pose
		[INFO] [WallTime: 1433359095.001546] Got new pose trajectory
5 sec >		[INFO] [WallTime: 1433359095.001821] Got new goal pose
		[WARN] [WallTime: 1433359095.002311] Received empty pose array. Clearing trajectory buffer
		[INFO] [WallTime: 1433359095.040784] MPC entered deadzone: pos 0.000129016458022 (0.01); orient 0.0274966244146 (10.0)

	B) rostopic echo /right/haptic_mpc/goal_pose displays:
		header: 
		  seq: 3
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
		  position: 
		    x: 0.5
		    y: -0.5
		    z: 0.0
		  orientation: 
		    x: 0.683012701892
		    y: 0.183012701892
		    z: -0.183012701892
		    w: 0.683012701892
		---
		header: 
		  seq: 4
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
		  position: 
		    x: 0.565443576665
		    y: -0.474902529147
		    z: 0.0285296053999
		  orientation: 
		    x: 0.685057477005
		    y: 0.103544253942
		    z: -0.210840774489
		    w: 0.689580313297
		---

4) Next iteration setOrientGoalRight([.5, -.5, 0], [0.707106781187, 0.0, 0.0, 0.707106781187], 20)

	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433359166.479650] Got new goal pose
		[INFO] [WallTime: 1433359166.479658] Updating MPC weights. Pos: 5.0, Orient: 5.0, Posture: 0.0
		[INFO] [WallTime: 1433359183.715757] Got new pose trajectory
		[INFO] [WallTime: 1433359183.716291] Got new goal pose
		[WARN] [WallTime: 1433359183.716810] Received empty pose array. Clearing trajectory buffer
		[INFO] [WallTime: 1433359183.760856] MPC entered deadzone: pos 0.000264667780625 (0.01); orient 0.0181103154298 (10.0)

	B) rostopic echo /right/haptic_mpc/goal_pose displays:
		header: 
		  seq: 5
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
		  position: 
		    x: 0.5
		    y: -0.5
		    z: 0.0
		  orientation: 
		    x: 0.707106781187
		    y: 0.0
		    z: 0.0
		    w: 0.707106781187
		---
		header: 
		  seq: 6
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
		  position: 
		    x: 0.50161111226
		    y: -0.501195368615
		    z: -0.0097481185099
		  orientation: 
		    x: 0.707490605468
		    y: -0.00086903939638
		    z: 0.0019546898095
		    w: 0.706719510933
		---

-----------------------------------------------------------------------------------
Demonstration of right arm procedure Test 3, AFTER CHANGING ALL TIMEOUTS TO 20 sec:
-----------------------------------------------------------------------------------
1) setPostureGoal([-1.570, 0, 0, -1.570, 3.141, 0, -4.712], 7)
	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433359999.834838] Updating MPC weights. Pos: 0.0, Orient: 0.0, Posture: 5.0
7 sec >		[INFO] [WallTime: 1433360007.001357] Got new pose trajectory
		[WARN] [WallTime: 1433360007.001817] Received empty pose array. Clearing trajectory buffer
		
	B) rostopic echo /right/haptic_mpc/goal_posture displays:
		header: 
		  seq: 1
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: ''
		data: [-1.57, 0.0, 0.0, -1.57, 3.141, 0.0, -4.712]
		---

2) setOrientGoalRight([.5, -.5, 0], [0.612372435696, 0.353553390593, -0.353553390593, 0.612372435696], 20)
	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433360077.342362] Updating MPC weights. Pos: 5.0, Orient: 5.0, Posture: 0.0
		[INFO] [WallTime: 1433360077.342916] Got new goal pose
		[INFO] [WallTime: 1433360098.002061] Got new pose trajectory
20 sec >	[INFO] [WallTime: 1433360098.002296] Got new goal pose
		[WARN] [WallTime: 1433360098.002731] Received empty pose array. Clearing trajectory buffer
		[INFO] [WallTime: 1433360098.040873] MPC entered deadzone: pos 8.37027216759e-05 (0.01); orient 0.0160662617645 (10.0)
		[INFO] [WallTime: 1433360139.320715] MPC entered deadzone: pos 0.00997973320662 (0.01); orient 1.51065297008 (10.0)

	B) rostopic echo /right/haptic_mpc/goal_pose displays:
		header: 
		  seq: 1
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
		  position: 
		    x: 0.5
		    y: -0.5
		    z: 0.0
		  orientation: 
		    x: 0.612372435696
		    y: 0.353553390593
		    z: -0.353553390593
		    w: 0.612372435696
		---
		header: 
		  seq: 2
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
		  position: 
		    x: 0.61774051699
		    y: -0.538097912955
		    z: -0.0492142085844
		  orientation: 
		    x: 0.616231166973
		    y: 0.283552195811
		    z: -0.317248984667
		    w: 0.662729494462
		---

3) Next iteration setOrientGoalRight([.5, -.5, 0], [0.683012701892, 0.183012701892, -0.183012701892, 0.683012701892], 20)
	A) start_pr2_mpc_all.launch displays:
		[INFO] [WallTime: 1433360282.705679] Updating MPC weights. Pos: 5.0, Orient: 5.0, Posture: 0.0
		[INFO] [WallTime: 1433360282.705726] Got new goal pose
		[INFO] [WallTime: 1433360303.001267] Got new pose trajectory
		[WARN] [WallTime: 1433360303.001692] Received empty pose array. Clearing trajectory buffer
20 sec >	[INFO] [WallTime: 1433360303.001919] Got new goal pose
		[INFO] [WallTime: 1433360303.040893] MPC entered deadzone: pos 2.49800180541e-16 (0.01); orient 0.0 (10.0)

		
	B) rostopic echo /right/haptic_mpc/goal_pose displays:
		header: 
		  seq: 3
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
		  position: 
		    x: 0.5
		    y: -0.5
		    z: 0.0
		  orientation: 
		    x: 0.683012701892
		    y: 0.183012701892
		    z: -0.183012701892
		    w: 0.683012701892
		---
		header: 
		  seq: 4
		  stamp: 
		    secs: 0
		    nsecs: 0
		  frame_id: /torso_lift_link
		pose: 
		  position: 
		    x: 0.558979995697
		    y: -0.470915524691
		    z: 0.0344183275524
		  orientation: 
		    x: 0.686344190835
		    y: 0.0856377889916
		    z: -0.229325576635
		    w: 0.684841295999
		---

4) Next iteration setOrientGoalRight([.5, -.5, 0], [0.707106781187, 0.0, 0.0, 0.707106781187], 20)

	A) start_pr2_mpc_all.launch displays:
		
	B) rostopic echo /right/haptic_mpc/goal_pose displays:
		

	

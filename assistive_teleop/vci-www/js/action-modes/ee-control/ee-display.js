var RFH = (function (module) {
    module.EEDisplay = function (options) {
        'use strict'; 
        self = this;
        var ros = options.ros;
        var side = options.side[0];
        var tfClient = options.tfClient;
        var localTFClient = new ROSLIB.TFClient({ros: ros,
            angularThres: 0.001,
            transThres: 0.001,
            rate: 10.0,
            fixedFrame: '/'+side+'_gripper_palm_link' });
        localTFClient.actionClient.cancel();

        var qx_rot = new THREE.Quaternion(1,0,0,0);  // Quaternion of 180 deg x-axis rotation, used to flip right finger models

        var currentGripper = new THREE.Object3D();
        currentGripper.userData.interactive = false;
        currentGripper.visible = false;
        var previewGripper = new THREE.Object3D();
        previewGripper.userData.interactive = false;
        previewGripper.visible = false;
        var goalGripper = new THREE.Object3D();
        goalGripper.userData.interactive = false;
        goalGripper.visible = false;
        var allGrippers = [currentGripper, previewGripper, goalGripper];
        RFH.viewer.scene.add(currentGripper, previewGripper, goalGripper);

        self.show = function () {
            currentGripper.visible = true;
        };

        self.showCurrent = function () {
            currentGripper.visible = true;
        };

        self.hideCurrent = function () {
            currentGripper.visible = false;
        };

        self.hide = function () {
            currentGripper.visible = false;
            previewGripper.visible = false;
            goalGripper.visible = false;
        };

        /*////////////  Load Gripper Model ////////////*/
        var colladaLoadProgress = function (data) {
            console.log("Loading Collada Mesh: ", data.loaded/data.total);
        };

        var gripperMaterial = new THREE.MeshBasicMaterial();
        gripperMaterial.transparent = true;
        gripperMaterial.depthTest = false;
        gripperMaterial.depthWrite = false;
        //gripperMaterial.color.setRGB(1.6,1.6,1.6); // Light gray default
        gripperMaterial.color.setRGB(2.4,2.4,0.2);
        gripperMaterial.opacity = 0.55;

        var previewMaterial = gripperMaterial.clone();
        previewMaterial.color.setRGB(1,0.5,0);
        previewMaterial.opacity = 0.4;

        var goalMaterial = gripperMaterial.clone();
        goalMaterial.color.setRGB(0.2, 3.0, 0.2);
        goalMaterial.opacity = 0.4;



        var updateRightFingerTF = function (tf) {
            for (var i=0; i<allGrippers.length; i += 1){
                var mesh = allGrippers[i].getObjectByName('rightFinger');
                mesh.position.set(tf.translation.x, tf.translation.y, tf.translation.z);
                var quat = new THREE.Quaternion(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
                quat = quat.multiplyQuaternions(quat, qx_rot);
                mesh.quaternion.set(quat.x, quat.y, quat.z, quat.w);
            }
            RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
        };

        var updateLeftFingerTF = function (tf) {
            for (var i=0; i<allGrippers.length; i += 1){
                var mesh = allGrippers[i].getObjectByName('leftFinger');
                mesh.position.set(tf.translation.x, tf.translation.y, tf.translation.z);
                mesh.quaternion.set(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
            }
            RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
        };

        var updateRightFingerTipTF = function (tf) {
            for (var i=0; i<allGrippers.length; i += 1){
                var mesh = allGrippers[i].getObjectByName('rightFingerTip');
                mesh.position.set(tf.translation.x, tf.translation.y, tf.translation.z);
                var quat = new THREE.Quaternion(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
                quat = quat.multiplyQuaternions(quat, qx_rot);
                mesh.quaternion.set(quat.x, quat.y, quat.z, quat.w);
            }
            RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
        };

        var updateLeftFingerTipTF = function (tf) {
            for (var i=0; i<allGrippers.length; i += 1){
                var mesh = allGrippers[i].getObjectByName('leftFingerTip');
                mesh.position.set(tf.translation.x, tf.translation.y, tf.translation.z);
                mesh.quaternion.set(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
            }
            RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
        };

        var palmOnLoad = function (collada) {
            // Create mesh with default values
            var palmGeom = collada.dae.geometries.palm3_M1000Shape.mesh.geometry3js;
            var palmMesh = new THREE.Mesh();
            palmMesh.name = 'palm';
            palmMesh.geometry = palmGeom;
            palmMesh.scale.set(0.1, 0.1, 0.1);
            // Set values for current pose display, add to currentGripper group
            palmMesh.material = gripperMaterial;
            currentGripper.add(palmMesh.clone());
            // Set values for undo preview pose display, add to previewGripper group
            palmMesh.material = previewMaterial;
            previewGripper.add(palmMesh.clone());
            // Set values for goal pose display, add to goalGripper group
            palmMesh.material = goalMaterial;
            goalGripper.add(palmMesh.clone());

            //        tfClient.subscribe(side+'_gripper_palm_link', updateCurrentGripperTF);
        };

        var fingerOnLoad = function (collada) {
            var fingerGeom = collada.dae.geometries.finger_M2K_modShape.mesh.geometry3js;
            var lFingerMesh = new THREE.Mesh();
            lFingerMesh.name = 'leftFinger';
            lFingerMesh.geometry = fingerGeom;
            lFingerMesh.scale.set(0.1, 0.1, 0.1);

            var rFingerMesh = new THREE.Mesh();
            rFingerMesh.name = 'rightFinger';
            rFingerMesh.geometry = fingerGeom;
            rFingerMesh.scale.set(0.1, 0.1, 0.1);

            lFingerMesh.material = gripperMaterial;
            rFingerMesh.material = gripperMaterial;
            currentGripper.add(lFingerMesh.clone(), rFingerMesh.clone());

            lFingerMesh.material = previewMaterial;
            rFingerMesh.material = previewMaterial;
            previewGripper.add(lFingerMesh.clone(), rFingerMesh.clone());

            lFingerMesh.material = goalMaterial;
            rFingerMesh.material = goalMaterial;
            goalGripper.add(lFingerMesh.clone(), rFingerMesh.clone());

            localTFClient.subscribe(side+'_gripper_l_finger_link', updateLeftFingerTF);
            localTFClient.subscribe(side+'_gripper_r_finger_link', updateRightFingerTF);
        };

        var fingerTipOnLoad = function (collada) {
            var fingerTipGeom =collada.dae.geometries.finger_tip_MShape.mesh.geometry3js.clone();
            var lFingerTipMesh = new THREE.Mesh();
            lFingerTipMesh.name = 'leftFingerTip';
            lFingerTipMesh.geometry = fingerTipGeom;
            lFingerTipMesh.scale.set(0.1, 0.1, 0.1);

            var rFingerTipMesh = new THREE.Mesh();
            rFingerTipMesh.name = 'rightFingerTip';
            rFingerTipMesh.geometry = fingerTipGeom;
            rFingerTipMesh.scale.set(0.1, 0.1, 0.1);

            lFingerTipMesh.material = gripperMaterial;
            rFingerTipMesh.material = gripperMaterial;
            currentGripper.add(lFingerTipMesh.clone(), rFingerTipMesh.clone());

            lFingerTipMesh.material = previewMaterial;
            rFingerTipMesh.material = previewMaterial;
            previewGripper.add(lFingerTipMesh.clone(), rFingerTipMesh.clone());

            lFingerTipMesh.material = goalMaterial;
            rFingerTipMesh.material = goalMaterial;
            goalGripper.add(lFingerTipMesh.clone(), rFingerTipMesh.clone());

            localTFClient.subscribe(side+'_gripper_l_finger_tip_link', updateLeftFingerTipTF);
            localTFClient.subscribe(side+'_gripper_r_finger_tip_link', updateRightFingerTipTF);
        };

        var gripperColladaLoader = new THREE.ColladaLoader();
        gripperColladaLoader.load('./data/gripper_model/gripper_palm.dae', palmOnLoad, colladaLoadProgress);
        gripperColladaLoader.load('./data/gripper_model/l_finger.dae', fingerOnLoad, colladaLoadProgress);
        gripperColladaLoader.load('./data/gripper_model/l_finger_tip.dae', fingerTipOnLoad, colladaLoadProgress);
        /*////////////  END Load Gripper Model ////////////*/

        var displayGoalPose = function (ps) {
            goalGripper.position.set(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z);
            goalGripper.quaternion.set(ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w);
            goalGripper.translateX(-0.18);
            goalGripper.visible = true;
            RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
        };

        var fullSide = side === 'r' ? 'right' : 'left';
        var handGoalSubscriber = new ROSLIB.Topic({
            ros: ros,
            name: fullSide + '_arm/haptic_mpc/goal_pose', 
            messageType: 'geometry_msgs/PoseStamped'
        });
//        handGoalSubscriber.subscribe(displayGoalPose);

        var deadzoneCB = function (bool_msg) {
            goalGripper.visible = !bool_msg.data;
        };
        var deadZoneSubscriber = new ROSLIB.Topic({
            ros: ros,
            name: fullSide+'_arm/haptic_mpc/in_deadzone',
            messageType: 'std_msgs/Bool'
        });
//        deadZoneSubscriber.subscribe(deadzoneCB);

        self.showPreviewGripper = function(ps) {
            previewGripper.position.set(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z);
            previewGripper.quaternion.set(ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w);
            previewGripper.translateX(-0.18); // Back up from tool_frame to gripper_palm_link
            previewGripper.visible = true;
        };

        self.hidePreviewGripper = function () {
            previewGripper.visible = false;
        };

        self.setCurrentPose = function (pose) {
            currentGripper.position.set(pose.position.x, pose.position.y, pose.position.z);
            currentGripper.quaternion.set(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
            currentGripper.translateX(-0.18);
            RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
        };
    };
    return module;
})(RFH || {});

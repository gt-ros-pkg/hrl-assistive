RFH.EEDisplay = function (options) {
    'use strict'; 
    self = this;
    var ros = options.ros;
    var side = options.side[0];
    var tfClient = options.tfClient;
    var localTFClient = new ROSLIB.TFClient({ros: ros,
                                            angularThres: 0.001,
                                            transThres: 0.001,
                                            rate: 10.0,
                                            fixedFrame: '/'+side'+gripper_palm_link' });
    localTFClient.actionClient.cancel();

    var qx_rot = new THREE.Quaternion(1,0,0,0);  // Quaternion of 180 deg x-axis rotation, used to flip right finger models

    self.show = function () {
        // TODO: Set all items as visible
    };

    self.hide = function () {
        //TODO: set all items NOT visible
    };
    
    /*////////////  Load Gripper Model ////////////*/
    var colladaLoadProgress = function (data) {
        console.log("Loading Collada Mesh: ", data.loaded/data.total);
    };

    var gripperMaterial = new THREE.MeshBasicMaterial();
    gripperMaterial.transparent = true;
    gripperMaterial.opacity = 0.25;
    gripperMaterial.depthTest = true;
    gripperMaterial.depthWrite = true;
    gripperMaterial.color.setRGB(1.6,1.6,1.6);

    var updateGripperPalmTF = function (tf) {
        palmMesh.position.set(tf.translation.x, tf.translation.y, tf.translation.z);
        palmMesh.quaternion.set(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
        RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
    };

    var updateGripperFingerTF = function (side, tf) {
        if (side === 'r') {
            rFingerMesh.position.set(tf.translation.x, tf.translation.y, tf.translation.z);
            var quat = new THREE.Quaternion(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
            quat = quat.multiplyQuaternions(quat, qx_rot);
            rFingerMesh.quaternion.set(quat.x, quat.y, quat.z, quat.w);
        } else {
            lFingerMesh.position.set(tf.translation.x, tf.translation.y, tf.translation.z);
            lFingerMesh.quaternion.set(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
        };
        RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
    };

    var updateGripperFingerTipTF = function (side, tf) {
        if (side === 'r') {
            rFingerTipMesh.position.set(tf.translation.x, tf.translation.y, tf.translation.z);
            var quat = new THREE.Quaternion(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
            quat = quat.multiplyQuaternions(quat, qx_rot);
            rFingerTipMesh.quaternion.set(quat.x, quat.y, quat.z, quat.w);
        } else {
            lFingerTipMesh.position.set(tf.translation.x, tf.translation.y, tf.translation.z);
            lFingerTipMesh.quaternion.set(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
        };
        RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);

    };
        
    var currentGrippger = new THREE.Object3D();
    var goalGripper = new THREE.Object3D();
    var undoGripper = new THREE.Object3D();

    var palmMesh = new THREE.Mesh();
    var lFingerMesh = new THREE.Mesh();
    var rFingerMesh = new THREE.Mesh();
    var lFingerTipMesh = new THREE.Mesh();
    var rFingerTipMesh = new THREE.Mesh();

    var palmOnLoad = function (collada) {
        // Set transforms + callback update
        var palmGeom = collada.dae.geometries.palm3_M1000Shape.mesh.geometry3js;
        palmMesh.geometry = palmGeom;
        palmMesh.material = gripperMaterial;
        palmMesh.scale.set(0.1, 0.1, 0.1);
        RFH.viewer.scene.add(palmMesh);
        tfClient.subscribe(side+'_gripper_palm_link', updateGripperPalmTF);
    }
    var fingerOnLoad = function (collada) {
        // Set transforms + callback update
        var fingerGeom = collada.dae.geometries.finger_M2K_modShape.mesh.geometry3js.clone();
        lFingerMesh = new THREE.Mesh(fingerGeom, gripperMaterial);
        lFingerMesh.scale.set(0.1, 0.1, 0.1);
        rFingerMesh = new THREE.Mesh(fingerGeom, gripperMaterial);
        rFingerMesh.scale.set(0.1, 0.1, 0.1);
        RFH.viewer.scene.add(rFingerMesh);
        RFH.viewer.scene.add(lFingerMesh);
        locaLTFClient.subscribe(side+'_gripper_l_finger_link', function(tf){updateGripperFingerTF('l', tf)});
        localTFClient.subscribe(side+'_gripper_r_finger_link', function(tf){updateGripperFingerTF('r', tf)});
    }
    var fingerTipOnLoad = function (collada) {
        // Set transforms + callback update
        var fingerTipGeom =collada.dae.geometries.finger_tip_MShape.mesh.geometry3js.clone();
        lFingerTipMesh = new THREE.Mesh(fingerTipGeom, gripperMaterial);
        lFingerTipMesh.scale.set(0.1, 0.1, 0.1);
        rFingerTipMesh = new THREE.Mesh(fingerTipGeom, gripperMaterial);
        rFingerTipMesh.scale.set(0.1, 0.1, 0.1);
        rFingerTipMesh.rotation.x = Math.PI;
        RFH.viewer.scene.add(lFingerTipMesh);
        RFH.viewer.scene.add(rFingerTipMesh);
        localTFClient.subscribe(side+'_gripper_l_finger_tip_link', function (tf){updateGripperFingerTipTF('l', tf)});
        localTFClient.subscribe(side+'_gripper_r_finger_tip_link', function (tf){updateGripperFingerTipTF('r', tf)});
    }

    var gripperColladaLoader = new THREE.ColladaLoader();
    gripperColladaLoader.load('./data/gripper_model/gripper_palm.dae', palmOnLoad, colladaLoadProgress);
    gripperColladaLoader.load('./data/gripper_model/l_finger.dae', fingerOnLoad, colladaLoadProgress);
    gripperColladaLoader.load('./data/gripper_model/l_finger_tip.dae', fingerTipOnLoad, colladaLoadProgress);
    /*////////////  END Load Gripper Model ////////////*/

    var displayGoalPose = function (ps_msg) {
        // TODO: Display new goal pose with shadow gripper
        RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
    };

    var handGoalSubscriber = new ROSLIB.Topic({
        ros: ros,
        name: side + '_arm/haptic_mpc/goal_pose', 
        messageType: 'geometry_msgs/PoseStamped'
    });
    handGoalSubscriber.subscribe(displayGoalPose);
}

var RFH = (function (module) {
    module.DriveZoneDisplay = function (options) {
        "use strict";
        var self = this;
        options = options || {};
        var ros = options.ros;
        var tfClient = options.tfClient;
        var $viewer = options.viewer;
        var goalPose = null;
        var odomCombinedTF = null;
        var goal_frame = 'odom_combined';

        // Init safe zone
        var zoneArea = new THREE.Mesh();
        zoneArea.visible = false;
        zoneArea.name = 'safeZone';
        var baseMaterial = new THREE.MeshBasicMaterial();
        baseMaterial.transparent = true;
        baseMaterial.depthTest = true;
        baseMaterial.depthWrite = false;
        baseMaterial.color.setRGB(0.0,1.2,0.0);
        baseMaterial.opacity = 0.65;
        zoneArea.material = baseMaterial;
        RFH.viewer.scene.add(zoneArea);

        self.show = function () {
            zoneArea.visible = true;    
        };

        self.hide = function () {
            zoneArea.visible = false;    
        };

        var updateGoalVisualization = function () {
            if (odomCombinedTF === null || zoneArea.geometry === null) {
                self.hide();
                return;
            } else {
                self.show();
            }

            var quat = new THREE.Quaternion(odomCombinedTF.rotation.x, odomCombinedTF.rotation.y, odomCombinedTF.rotation.z, odomCombinedTF.rotation.w);
            var odomCombinedMat = new THREE.Matrix4();
            odomCombinedMat.makeRotationFromQuaternion(quat);
            odomCombinedMat.setPosition(new THREE.Vector3(odomCombinedTF.translation.x, odomCombinedTF.translation.y, odomCombinedTF.translation.z));

            var goalQuat = new THREE.Quaternion(goalPose.pose.orientation.x, goalPose.pose.orientation.y, goalPose.pose.orientation.z, goalPose.pose.orientation.w);
            var goalMat = new THREE.Matrix4();
            goalMat.makeRotationFromQuaternion(goalQuat);
            goalMat.setPosition(new THREE.Vector3(goalPose.pose.position.x, goalPose.pose.position.y, goalPose.pose.position.z));

            odomCombinedMat.multiply(goalMat);
            var trans = new THREE.Matrix4();
            var scale = new THREE.Vector3();
            odomCombinedMat.decompose(trans, quat, scale);
            zoneArea.position.set(trans.x, trans.y, trans.z);
            baseModel.quaternion.set(quat.x, quat.y, quat.z, quat.w);
            RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
        };

        var updateGeom = function (paMsg) {
            // Create mesh with default values
            // Clear old Geom?
            // Set box base on PA points + math
            var allPoints = new Array(4);
            var centerPoint = [0.0, 0.0, 0.0];
            for (i = 0; i < 4; i++){
               allPoints[i] = paMsg.poses[0].position;
            }
            for (j = 0; j < allPoints[0].length; j++){
               for (k = 0; k < allPoints.length; k++){
                   centerPoint[j] += allPoints[k][j];
               }
               centerPoint[j] /= 4;
            }
            goalPose.position.x = centerPoint[0];
            goalPose.position.y = centerPoint[1];
            goalPose.position.z = 0;
            goalPose.orientation = paMsg.poses[0].orientation;
            var baseGeom = new THREE.BoxGeometry(700, 700, 10 );
            zoneArea.geometry = baseGeom;
            zoneArea.scale.set(0.1, 0.1, 0.1);
        };

        var goalSub = new ROSLIB.Topic({
            ros: ros,
            name: 'move_back_safe_zone',
            messageType: 'geometry_msgs/PoseArray'
        });

        var setGoal = function (paMsg) {
            updateGeom(paMsg);
            updateGoalVisualization();
        };
        goalSub.subscribe(setGoal);

        self.clearGoal = function () {
            zoneArea.geometry = null;
            updateGoalVisualization();
        };

        var updateOdomTF = function (tf) {
            odomCombinedTF = tf;
            updateGoalVisualization();
        };
        tfClient.subscribe(goal_frame, updateOdomTF);


    };
    return module;

})(RFH || {});

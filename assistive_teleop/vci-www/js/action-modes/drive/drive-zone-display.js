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
        var zoneArea = new THREE.Object3D();
        zoneArea.visible = false;
        RFH.viewer.scene.add(zoneArea);
        var goal_frame = 'odom_combined';

        self.show = function () {
            zoneArea.visible = true;    
        };

        self.hide = function () {
            zoneArea.visible = false;    
        };

        var updateGoalVisualization = function () {
            if (odomCombinedTF === null || goalPose === null) {
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

        var goalSub = new ROSLIB.Topic({
            ros: ros,
            name: 'move_back_safe_zone',
            messageType: 'geometry_msgs/PoseArray'
        });

        var setGoal = function (psMsg) {
            goalPose = psMsg;
            updateGoalVisualization();
            loadZone();
        };
        goalSub.subscribe(setGoal);

        self.clearGoal = function () {
            goalPose = null;
            updateGoalVisualization();
        };

        var updateOdomTF = function (tf) {
            odomCombinedTF = tf;
            updateGoalVisualization();
        };

        var loadZone = function () {
            // Create mesh with default values
            var baseGeom = new THREE.BoxGeometry(700, 700, 700, 10, 10, 10);
            var baseMesh = new THREE.Mesh();
            baseMesh.name = 'base';
            baseMesh.geometry = baseGeom;
            baseMesh.scale.set(0.1, 0.1, 0.1);

            var baseMaterial = new THREE.MeshBasicMaterial();
            baseMaterial.transparent = true;
            baseMaterial.depthTest = true;
            baseMaterial.depthWrite = false;
            baseMaterial.color.setRGB(0.0,1.2,0.0);
            baseMaterial.opacity = 0.65;

            baseMesh.material = baseMaterial;
            zoneArea.add(baseMesh);
            tfClient.subscribe(goal_frame, updateOdomTF);
        };

    };
    return module;

})(RFH || {});

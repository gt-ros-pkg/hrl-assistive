var RFH = (function (module) {
    module.DriveZoneDisplay = function (options) {
        "use strict";
        var self = this;
        options = options || {};
        var ros = options.ros;
        var tfClient = options.tfClient;
        var $viewer = options.viewer;
        var goalPose = null;
        var goalWidth = 0.0;
        var goalLength = 0.0;
        var goal_frame = 'odom_combined';
        var odomCombinedTF = null;

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
            zoneArea.quaternion.set(quat.x, quat.y, quat.z, quat.w);
            RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
        };

        var updateGeom = function (paMsg) {
            // Create mesh with default values
            // Clear old Geom?
            // Set box base on PA points + math
            var allPoints = [];
            var centerPoint = [0.0, 0.0, 0.0];
            for (var i = 0; i < 4; i += 1){
               allPoints.push([paMsg.poses[i].position.x, 
                               paMsg.poses[i].position.y, 
                               paMsg.poses[i].position.z]);
            }
            for (var j = 0; j < allPoints[0].length; j += 1){
               for (var k = 0; k < allPoints.length; k += 1){
                   centerPoint[j] += allPoints[k][j];
               }
               centerPoint[j] /= 4;
            }

            goalPose = ros.composeMsg('geometry_msgs/PoseStamped'); // Assign to goalPose from outer scope (init'ed to null above), so all fn's can access the data
            goalPose.header.frame_id = goal_frame; // Gets goal frame from outer scope (where set at top of file)
            goalPose.pose.position.x = centerPoint[0];
            goalPose.pose.position.y = centerPoint[1];
            goalPose.pose.position.z = 0;
            goalPose.pose.orientation = paMsg.poses[0].orientation;

            var baseGeom = new THREE.BoxGeometry(goalWidth.data, goalLength.data, 0.1 );
            //var baseGeom = new THREE.BoxGeometry(10, 10, 10 );
            zoneArea.geometry = baseGeom;
            zoneArea.scale.set(1, 1, 0.1);
        };

        var widthSub= new ROSLIB.Topic({
            ros: ros,
            name: 'move_back_safe_zone/width',
            messageType: 'std_msgs/Float32'
        });

        var setWidth= function (wid) {
            // Don't create goalpose here, just let it update the variable in the outer scope (instantiated at top of file)
            goalWidth = wid;
            updateGoalVisualization();  // Refreshes visualization to reflect updated zone pose and geometery
        };
        widthSub.subscribe(setWidth);


        var lengthSub = new ROSLIB.Topic({
            ros: ros,
            name: 'move_back_safe_zone/length',
            messageType: 'std_msgs/Float32'
        });

        var setLength = function (len) {
            // Don't create goalpose here, just let it update the variable in the outer scope (instantiated at top of file)
            goalLength = len; // Updates goal pose and geometry
            updateGoalVisualization();  // Refreshes visualization to reflect updated zone pose and geometery
        };
        lengthSub.subscribe(setLength);

        var goalSub = new ROSLIB.Topic({
            ros: ros,
            name: 'move_back_safe_zone/points',
            messageType: 'geometry_msgs/PoseArray'
        });

        var setGoal = function (paMsg) {
            // Don't create goalpose here, just let it update the variable in the outer scope (instantiated at top of file)
            updateGeom(paMsg); // Updates goal pose and geometry
            updateGoalVisualization();  // Refreshes visualization to reflect updated zone pose and geometery
        };
        goalSub.subscribe(setGoal);

        self.clearGoal = function () {
            goalPose = null; // If no goal pose, updateGoalVisualiztion will hide the rendered zone object
            updateGoalVisualization();
        };

        var updateOdomTF = function (tf) {
            odomCombinedTF = tf;  // Update the tf stored in outer scope
            updateGoalVisualization();  // refresh rendering (to use new tf)
        };
        tfClient.subscribe(goal_frame, updateOdomTF);
    };
    return module;

})(RFH || {});

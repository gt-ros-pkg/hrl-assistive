RFH.DriveGoalDisplay = function (options) {
    "use strict";
    var self = this;
    options = options || {};
    var ros = options.ros;
    var tfClient = options.tfClient;
    var $viewer = options.viewer;
    var goalPose = null;
    var odomCombinedTF = null;
    var baseModel = new THREE.Object3D();
    RFH.viewer.scene.add(baseModel);

    self.show = function () {
        baseModel.visible = true;    
    };

    self.hide = function () {
        baseModel.visible = false;    
    };

    var updateGoalVisualization = function () {
        if (odomCombinedTF === null || goalPose === null) {
            self.hide();
            return;
        } else {
            self.show();
        }

        var quat = new THREE.Quaternion(odomCombinedTF.rotation.x, odomCombinedTF.rotation.y, odomCombinedTF.rotation.z, odomCombinedTF.rotation.w);
        var odomCombinedMat = new THREE.Matrix4()
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
        baseModel.position.set(trans.x, trans.y, trans.z);
        baseModel.quaternion.set(quat.x, quat.y, quat.z, quat.w);
        RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
    };

    var goalSub = new ROSLIB.Topic({
        ros: ros,
        name: 'servo_open_loop_goal',
        messageType: 'geometry_msgs/PoseStamped'
    });

    var setGoal = function (psMsg) {
        goalPose = psMsg;
        updateGoalVisualization();
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
    
    var colladaLoadProgress = function (data) {
        console.log("Loading Base Collada Mesh: ", 100*data.loaded/data.total, "%" );
    };

    var baseOnLoad = function (collada) {
        // Create mesh with default values
        var baseGeom = collada.dae.geometries['shape0-lib'].mesh.geometry3js;
        var baseMesh = new THREE.Mesh();
        baseMesh.name = 'base';
        baseMesh.geometry = baseGeom;
        baseMesh.scale.set(0.1, 0.1, 0.1);

        var baseMaterial = new THREE.MeshBasicMaterial();
        baseMaterial.transparent = true;
        baseMaterial.depthTest = true;
        baseMaterial.depthWrite = false;
        baseMaterial.color.setRGB(1.2,1.2,1.2);
        baseMaterial.opacity = 0.65;

        baseMesh.material = baseMaterial;
        baseModel.add(baseMesh);
        tfClient.subscribe('odom_combined', updateOdomTF);
    };

    var colladaLoader = new THREE.ColladaLoader();
    colladaLoader.load('./data/base_model/base_simplified.dae', baseOnLoad, colladaLoadProgress);
};

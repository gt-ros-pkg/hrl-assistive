RFH.ParamLocation = function(options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    var paramName = options.paramName || 'id_location';
    self.name = options.name || 'paramLocationTask';
    self.showButton = false;
    self.container = options.container;
    self.camera = options.camera;
    var positionOverride = null;
    var orientationOverride = null;
    var offset = {position:{x:0, y:0, z:0},
                   rotation:{x:0, y:0, z:0}};
    self.pixel23d = new RFH.Pixel23DClient({
        ros: ros,
        cameraInfoTopic: self.camera.infoTopic
    });
//    self.buttonText = 'ID_Location';
//    self.buttonClass = 'id-location-button';
    self.$edges = $('.map-look');
    self.$image = $('#mjpeg-image');

    self.setParam = function(paramName) {
        self.param = new ROSLIB.Param({
            ros: ros,
            name: paramName
        });
    };
    self.setParam(options.paramName);

    self.getParamName = function () {
        return self.param.name;
    };

    self.getParam = function (cb) {
        self.param.get(cb);
    };

    self.getOffset = function () {
        return offset;
    };

    self.setOffset = function (new_offset) {
        try {
            offset.position.x = new_offset.position.x || offset.position.x;
            offset.position.y = new_offset.position.y || offset.position.y;
            offset.position.z = new_offset.position.z || offset.position.z;
        } catch (err) {
            offset.position = {'x':0, 'y':0, 'z':0};
        }
        try {
            offset.rotation.x = new_offset.rotation.x || offset.rotation.x;
            offset.rotation.y = new_offset.rotation.y || offset.rotation.y;
            offset.rotation.z = new_offset.rotation.z || offset.rotation.z;
            offset.rotation.w = new_offset.rotation.w || offset.rotation.w;
        } catch (err) {
            offset.rotation = {'x':0, 'y':0, 'z':0, 'w':1};
        }
    };

    self.setOrientationOverride = function (quat) {
            orientationOverride = quat;
    };

    self.setPositionOverride = function (position) {
        positionOverride = position;
    };

    self.poseCB = function(pose_msg) {
        console.log("Original Pose msg: ", pose_msg);
        self.$image.removeClass('cursor-wait');
        pose_msg.pose = applyOffset(pose_msg.pose);
        console.log("Modified msg: ", pose_msg);
        //var pose_dict = pose_msg_to_dict(pose_msg);
        self.param.set(pose_msg);
//        RFH.regions.push(new RFH.RegionView({ros: ros,
//                                             name: self.param.name,
//                                             viewer: RFH.viewer,
//                                             center: pose_msg.pose.position,
//                                             radius: 0.2}));
                                                        
    };

    self.clickCB = function(event, ui) {
        var pt = RFH.positionInElement(event);
        var px = (pt[0]/event.target.clientWidth);
        var py = (pt[1]/event.target.clientHeight);
        try {
            self.pixel23d.callRelativeScale(px, py, self.poseCB);
            self.$image.addClass('cursor-wait');
        } catch(err) {
            log(err);
        }
    };

    self.start = function() {
        self.$edges.addClass('visible').show();
        self.$image.addClass('cursor-select');
        self.$image.on('click.id-location', self.clickCB);
    };

    self.stop = function() {
        self.$edges.removeClass('visible').hide();
        self.$image.removeClass('cursor-select');
        self.$image.off('click.id-location');
    };

    var applyOffset = function (pose) {
        var quat = new THREE.Quaternion(pose.orientation.x,
                                        pose.orientation.y,
                                        pose.orientation.z,
                                        pose.orientation.w);
        var poseRotMat = new THREE.Matrix4().makeRotationFromQuaternion(quat);
        var offsetVec = new THREE.Vector3(offset.position.x, 
                                       offset.position.y,
                                       offset.position.z); //Get to x dist from point along normal
        offsetVec.applyMatrix4(poseRotMat);
        var desRotMat = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(offset.rotation.x,
                                                                                  offset.rotation.y,
                                                                                  offset.rotation.z));
        poseRotMat.multiply(desRotMat);
        poseRotMat.setPosition(new THREE.Vector3(pose.position.x + offsetVec.x,
                                                 pose.position.y + offsetVec.y,
                                                 pose.position.z + offsetVec.z));
        var trans = new THREE.Matrix4();
        var scale = new THREE.Vector3();
        poseRotMat.decompose(trans, quat, scale);
        pose.position.x = trans.x;
        pose.position.y = trans.y;
        pose.position.z = trans.z;
        pose.orientation.x = quat.x;
        pose.orientation.y = quat.y;
        pose.orientation.z = quat.z;
        pose.orientation.w = quat.w;
        if (positionOverride !== null){
            pose.position.x = positionOverride.x;
            pose.position.y = positionOverride.y;
            pose.position.z = positionOverride.z;
        }
        if (orientationOverride !== null){
            pose.orientation.x = orientationOverride.x;
            pose.orientation.y = orientationOverride.y;
            pose.orientation.z = orientationOverride.z;
            pose.orientation.w = orientationOverride.w;
        }
        return pose;
    };

};

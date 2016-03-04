RFH.PublishLocation = function(options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.name = options.name || 'idLocationTask';
    self.topic = options.topic || 'id_location';
    self.camera = options.camera;
    self.container = options.container;
    var offset = {position:{x:0, y:0, z:0},
                   rotation:{x:0, y:0, z:0}};
    self.pixel23d = new RFH.Pixel23DClient({
        ros: self.ros,
        cameraInfoTopic: self.camera.infoTopic
    });
//    self.buttonText = 'ID_Location';
 //   self.buttonClass = 'id-location-button';
    self.$edges = $('.map-look');
    self.$image = $('#mjpeg-image');

    self.ros.getMsgDetails('geometry_msgs/PoseStamped');
    self.posePublisher = new ROSLIB.Topic({
        ros: self.ros,
        name: '/id_location',
        messageType: 'geometry_msgs/PoseStamped'
    });
    self.posePublisher.advertise();

    self.getOffset = function () {
        return offset;
    };

    self.setOffset = function (new_offset) {
        offset.position.x = new_offset.position.x || offset.position.x;
        offset.position.y = new_offset.position.y || offset.position.y;
        offset.position.z = new_offset.position.z || offset.position.z;
        offset.rotation.x = new_offset.rotation.x || offset.rotation.x;
        offset.rotation.y = new_offset.rotation.y || offset.rotation.y;
        offset.rotation.z = new_offset.rotation.z || offset.rotation.z;
        offset.rotation.w = new_offset.rotation.w || offset.rotation.w;
    };

    self.poseCB = function(pose_msg) {
        console.log("Original Pose msg: ", pose_msg);
        self.$image.removeClass('cursor-wait');
        pose_msg.pose = applyOffset(pose_msg.pose);
        console.log("Moidified msg: ", pose_msg);
        self.posePublisher.publish(pose_msg);
        console.log("Pixel23D Returned");
    };

    self.clickCB = function(event, ui) {
        var pt = RFH.positionInElement(event);
        var px = (pt[0]/event.target.clientWidth);
        var py = (pt[1]/event.target.clientHeight);
        try {
            self.pixel23d.callRelativeScale(px, py, self.poseCB);
            console.log("Called Pixel23D");
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
        return pose;
    };

};

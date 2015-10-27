RFH.IdLocation = function(options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.name = options.name || 'idLocationTask';
    self.container = options.container;
    self.offset = {position:{x:0, y:0, z:0},
                   rotation:{x:0, y:0, z:0}};
    self.pixel23d = new RFH.Pixel23DClient({
        ros: self.ros,
        cameraInfoTopic: '/head_mount_kinect/rgb_lowres/camera_info'
    });
//#    self.buttonText = 'ID_Location';
//    self.buttonClass = 'id-location-button';
    self.$edges = $('.map-look');
    self.$image = $('#mjpeg-image');

    self.ros.getMsgDetails('geometry_msgs/PoseStamped');
    self.posePublisher = new ROSLIB.Topic({
        ros: self.ros,
        name: '/id_location',
        messageType: 'geometry_msgs/PoseStamped'
    });
    self.posePublisher.advertise();

    self.poseCB = function(pose_msg) {
        self.$image.removeClass('cursor-wait');
//        pose_msg.pose = applyOffset(pose_msg.pose);
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
        var offset = new THREE.Vector3(self.offset.position.x, self.offset.position.y, self.offset.position.z); //Get to x dist from point along normal
        offset.applyMatrix4(poseRotMat);
        var desRotMat = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(self.offset.rotation.x, self.offset.rotation.y, self.offset.rotation.z));
        poseRotMat.multiply(desRotMat);
        poseRotMat.setPosition(new THREE.Vector3(pose.position.x + offset.x,
                                                 pose.position.y + offset.y,
                                                 pose.position.z + offset.z));
        var trans = new THREE.Matrix4();
        var scale = new THREE.Vector3();
        poseRotMat.decompose(trans, quat, scale);
        pose.position.x = trans[0];
        pose.position.y = trans[1];
        pose.position.z = trans[2];
        pose.orientation.x = quat.x;
        pose.orientation.y = quat.y;
        pose.orientation.z = quat.z;
        pose.orientation.w = quat.w;
        return pose;
    };

};

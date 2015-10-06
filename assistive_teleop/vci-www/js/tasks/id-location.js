RFH.IdLocation = function(options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.name = options.name || 'idLocationTask';
    self.container = options.container;
    self.pixel23d = new RFH.Pixel23DClient({
        ros: self.ros,
        cameraInfoTopic: '/head_mount_kinect/rgb_lowres/camera_info'
    });
    self.buttonText = 'ID_Location';
    self.buttonClass = 'id-location-button';
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
};

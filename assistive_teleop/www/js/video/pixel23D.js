RFH.Pixel23DClient = function (options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.cameraInfoTopic = options.cameraInfoTopic;
    self.imageHeight = null;
    self.imageWidth = null;
    self.serviceName = options.serviceName || '/pixel_2_3d';

    self.updateCameraInfo = function (camInfoMsg) {
        self.imageHeight = camInfoMsg.height;
        self.imageWidth = camInfoMsg.width;
    };

    self.ros.getMsgDetails('sensor_msgs/CameraInfo');
    self.cameraInfoSubscriber = new ROSLIB.Topic({
        ros: self.ros,     
        name: self.cameraInfoTopic,
        messageType: 'sensor_msgs/CameraInfo'});
    self.cameraInfoSubscriber.subscribe(self.updateCameraInfo);

    self.serviceClient =  new ROSLIB.Service({
                                        ros: self.ros,
                                        name: self.serviceName,
                                        serviceType: 'Pixel23d'});

    // Calls the service with given pixel coordinated,
    // checks for success, and calls given callback with returned pose
    self.call = function (u, v, cb) {
        var req = new ROSLIB.ServiceRequest({'pixel_u':u, 'pixel_v':v});
        var cb_err_wrap = function (resp) {
            switch (resp.error_flag) {
                case 0:
                    cb(resp.pixel3d.pose);
                    break;
                case 1:
                    throw "Pixel23D (u: %u%, v:%v%): No Camera Info Received".replace("%u%", u).replace("%v%", v);
                    break;
                case 2:
                    throw "Pixel23D (u: %u%, v:%v%): No Pointcloud Received".replace("%u%", u).replace("%v%", v);
                    break;
                case 3:
                    throw "Pixel23D (u: %u%, v:%v%): Requested pixel is outside image".replace("%u%", u).replace("%v%", v);
                    break;
                case 4:
                    throw "Pixel23D (u: %u%, v:%v%): No Pointcloud data at requested pixel".replace("%u%", u).replace("%v%", v);
                    break;

            };
        };
        self.serviceClient.callService(req, cb_err_wrap);
    };

    // Receives pixel values in 0-1 range, scales to size of image used by pixel23d, and calls.
    self.callRelativeScale = function (u, v, cb) {
        u *= self.imageWidth;
        v *= self.imageHeight;
        self.call(u, v, cb);
    };
}

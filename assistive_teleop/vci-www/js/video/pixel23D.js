var RFH = (function (module) {
    'use strict';
    module.Pixel23DClient = function (options) {
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
                console.log("Pixel23D response:\n");
                console.log(resp);
                switch (resp.error_flag) {
                    case 0:
                        cb(resp.pixel3d);
                        break;
                    case 1:
                        RFH.log("ERROR: Still waiting for 3D camera image data");
                        cb(null);
                        throw "Pixel23D (u: %u%, v:%v%): No Camera Info Received".replace("%u%", u).replace("%v%", v);
                    case 2:
                        RFH.log("ERROR: Still waiting for 3D camera depth data");
                        cb(null);
                        throw "Pixel23D (u: %u%, v:%v%): No Pointcloud Received".replace("%u%", u).replace("%v%", v);
                    case 3:
                        RFH.log("ERROR: Invalid Location Requested.");
                        cb(null);
                        throw "Pixel23D (u: %u%, v:%v%): Requested pixel is outside image".replace("%u%", u).replace("%v%", v);
                    case 4:
                        RFH.log("ERROR: No Depth Data available at selected location.");
                        cb(null);
                        throw "Pixel23D (u: %u%, v:%v%): No Pointcloud data at requested pixel".replace("%u%", u).replace("%v%", v);
                }
            };
            self.serviceClient.callService(req, cb_err_wrap);
        };

        // Receives pixel values in 0-1 range, scales to size of image used by pixel23d, and calls.
        self.callRelativeScale = function (u, v, cb) {
            u = Math.round(u*self.imageWidth);
            v = Math.round(v*self.imageHeight);
            self.call(u, v, cb);
        };
    };
    return module;
})(RFH || {});

RFH.Pixel23DClient = function (ros) {
    'use strict';
    var self = this;
    self.ros = ros;
    self.serviceClient =  new ROSLIB.Service({
                                        ros: self.ros,
                                        name: '/pixel_2_3d',
                                        serviceType: 'Pixel23d'});
    self.call = function (u, v, cb) {
        var req = new self.ros.ServiceRequest({'pixel_u':u, 'pixel_v':v});
        self.serviceClient.callService(req, cb);
    }
}

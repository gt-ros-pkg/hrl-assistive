RFH.Drive = function (options) {
    "use strict";
    var self = this;
    var options = options || {};
    self.ros = options.ros;
    self.div = options.div || 'markers';
    self.tfClient = options.tfClient;
    self.camera = options.camera;
    self.buttonText = 'Drive';
    self.buttonClass = 'drive-button';
    self.headTF = new ROSLIB.Transform()
    
    self.updateHead = function (transform) { self.headTF = transform; }
    self.tfClient.subscribe(options.camera.frame_id, self.updateHead);

    self.start = function () {
        $('#'+self.div+' canvas').on('click.rfh', self.onClick);
    }

    self.stop = function () {
        $('#'+self.div+' canvas').off('click.rfh');
    }

    self.onClick = function (e) {
        var pt = RFH.positionInElement(e); 
        var px = (pt[0]/e.target.clientWidth) * self.camera.width;
        var py = (pt[1]/e.target.clientHeight) * self.camera.height;
        if (self.camerea.frame_id === '') {
            alert("Camera position not up to date.  Cannot drive safely.");
            self.camera.updateCameraInfo();
            }
        var xyz = self.camera.projectPixel(px, py, 1.0);
        var pose = new ROSLIB.Pose({position:{x: xyz[0],
                                              y: xyz[1], 
                                              z: xyz[2]}});
        pose.applyTransform(self.headTF);
        console.log("Projected Pose in Base Link");
        console.log(pose);
    }

}

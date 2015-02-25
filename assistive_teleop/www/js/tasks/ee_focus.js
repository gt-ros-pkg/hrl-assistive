RFH.FocalPoint = function (options) {
    "use strict";
    var self = this;
    self.point = null;
    self.side = options.side;
    self.divId = options.divId+"-image";
    self.pointDivId = options.pointDivId || self.side+'FocusPoint';
    self.tfClient = options.tfClient;
    self.camera = options.camera;
    self.pointDiv = options.pointDiv;
    $('#'+self.pointDivId).hide(); // Start off with point hidden

    self.positionFocusPointImage = function (trans) {
        if (self.point === null) { return };
        var pixel = self.camera.projectPoint(self.point.x, self.point.y, self.point.z, 'base_link');
        var u = pixel[0] + $('#'+self.pointDivId).width;
        var v = pixel[1] + $('#'+self.pointDivId).height;
        $('#'+self.pointDivId).css({'left':u, 'top':v});
    };

    self.pixel23d = new RFH.Pixel23DClient({
            ros: RFH.ros,
            cameraInfoTopic: '/head_mount_kinect/rgb_lowres/camera_info',
            serviceName: '/pixel_2_3d'
        });

    self.setFocusPoint = function (pose) {
        self.point = new THREE.Vector3(pose.position.x,
                                       pose.position.y,
                                       pose.position.z);
        self.tfClient.subscribe(self.camera.frame_id, self.positionFocusPointImage);
        $('#'+self.pointDivId).show();
        $('#select-focus-toggle').removeAttr('checked').button("refresh");
    };

    self.getNewFocusPoint = function () {
        var oldCursor = $('#'+self.divId).css('cursor');
        $('#'+self.divId).css('cursor', 'url(./css/cursors/focus/focus-pointer.png) 15 60, auto');
        var clickCB = function (e) {
            e.stopPropagation();
            var pt = RFH.positionInElement(e);
            var x = pt[0]/self.camera.width;
            var y = pt[1]/self.camera.height;
            $('#'+self.divId).css('cursor', oldCursor);
            self.pixel23d.callRelativeScale(x, y, self.setFocusPoint);
        };
        $('#'+self.divId).one('click', clickCB);
    };

    self.clear = function () {
        self.tfClient.unsubscribe(self.camera.frame_id, self.positionFocusPointImage);
        $('#'+self.pointDivId).hide();
        self.point = null;
    };

};

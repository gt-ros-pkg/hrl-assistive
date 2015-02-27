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
        var pixel = self.camera.projectPoint(self.point.x,
                                             self.point.y,
                                             self.point.z,
                                             'base_link');
        var u = pixel[0];
        var v = pixel[1];
        $('#'+self.pointDivId).css({'left':u, 'top':v});
    };

    self.pixel23d = new RFH.Pixel23DClient({
            ros: RFH.ros,
            //cameraInfoTopic: '/head_mount_kinect/rgb_lowres/camera_info',
            cameraInfoTopic: '/head_mount_kinect/rgb/camera_info',
            serviceName: '/pixel_2_3d'
        });

    self.setFocusPoint = function (pose) {
        self.point = new THREE.Vector3(pose.position.x,
                                       pose.position.y,
                                       pose.position.z);
        self.tfClient.subscribe(self.camera.frame_id, self.positionFocusPointImage);
        $('#'+self.pointDivId).show();
        $('#select-focus-toggle').removeAttr('checked').button("refresh");
        $('.depth-mask').hide();
    };


    self.getNewFocusPoint = function () {
        var oldCursor = $('#'+self.divId).css('cursor');
        $('#'+self.divId).css('cursor', 'url(./css/cursors/focus/focus-pointer.png) 20 8, auto');
        $(".map-look").hide();
//        $('.depth-mask').show();
        $('#'+self.divId).on('mousemove', function(e) { 
                                            console.log(RFH.positionInElement(e));
                                            });
        var clickCB = function (e) {
            e.stopPropagation();
            var pt = RFH.positionInElement(e);
            var lMaskWidth = $('#depthMaskLeft').width(); 
            var rMaskWidth = $('#depthMaskRight').width(); 
//            var x = (pt[0] - lMaskWidth) / (self.camera.width - lMaskWidth - rMaskWidth);
            var x = (pt[0]/e.target.width);
            var y = (pt[1]/e.target.height);
            $('#'+self.divId).css('cursor', oldCursor);
            $(".map-look").show();
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

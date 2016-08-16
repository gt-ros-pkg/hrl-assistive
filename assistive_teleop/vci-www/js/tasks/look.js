RFH.Look = function (options) {
    'use strict';
    var self = this;
    self.name = options.name || 'lookingTask';
    var ros = options.ros;
    self.showButton = true;
    self.buttonText = "Look";
    self.toolTipText = "Move the head to look around";
    var imageDivId = options.imageDivId || 'mjpeg-image';
    var $imageDiv = $("#" + imageDivId);
    self.mapLookDivs = $(".map-look");
    self.camera = options.camera || new RFH.ROSCameraModel();
    self.head = options.head || new Pr2Head(ros);
    var zoomLevel = 1.0;
    var maxZoom = 4;
    self.zoomServiceClient = new ROSLIB.Service({
        ros: ros,
        name: '/set_cropdecimate',
        serviceType: 'assistive_teleop/SetCropDecimateParams'
    });

    var hfov = 1;
    var vfov = 0.75; //FOV of kinect is ~1 radians wide, 0.75 radians tall
    var SCALE = 0.8; //Scale large motions so we don't over shoot
    var edgeLook = function (event) {
        var dx = 0, dy = 0;
        var classes = event.target.classList;
        if (classes.contains("top")) { dy = -SCALE  * vfov/zoomLevel; }
        if (classes.contains("bottom")) { dy = SCALE  * vfov/zoomLevel; }
        if (classes.contains("left")) { dx = SCALE * hfov/zoomLevel; }
        if (classes.contains("right")) { dx = -SCALE * hfov/zoomLevel; }
        self.head.delPosition(dx, dy); 
        event.stopPropagation();
    };
    var lookAreas = $('.map-look').on('click.rfh-look', edgeLook);

    self.pointHead = function (e) {
        var pt = RFH.positionInElement(e); 
        var pctOffset = (50 - (50/zoomLevel))/100;
        var px = ((pt[0]/e.target.clientWidth)/zoomLevel + pctOffset) * self.camera.width;
        var py = ((pt[1]/e.target.clientHeight)/zoomLevel + pctOffset) * self.camera.height;
        var xyz =  self.camera.projectPixel(px, py);
        self.head.pointHead(xyz[0], xyz[1], xyz[2], self.camera.frame_id);
    };

    var zoomIn = function (e) {
        zoomLevel += 1;
        $zoomOutButton.button('enable');
        if (zoomLevel >= maxZoom) { $zoomInButton.button('disable'); };
        $imageDiv.css({transform:'scale('+zoomLevel+')'});
    };
    var $zoomInButton = $('#controls > .zoom.in').button().on('click.rfh', zoomIn)

    var zoomOut = function (e) {
        zoomLevel -= 1;
        $zoomInButton.button('enable');;
        if (zoomLevel <= 1) { $zoomOutButton.button('disable'); };
        $imageDiv.css({transform:'scale('+zoomLevel+')'});
    };
    var $zoomOutButton = $('#controls > .zoom.out').button().on('click.rfh', zoomOut).button('disable');

    self.start = function () {
        $imageDiv.addClass("cursor-eyes").on("click.rfh-look", self.pointHead);
        self.mapLookDivs.css("display","block");
        console.log('Looking task started');
    };

    self.stop = function () {
        $imageDiv.removeClass("cursor-eyes").off("click.rfh-look");
        self.mapLookDivs.css("display","none");
        console.log('Looking task stopped');
    };
};

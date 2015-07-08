RFH.Look = function (options) {
    'use strict';
    var self = this;
    self.name = options.name || 'lookingTask';
    self.ros = options.ros;
    self.div = options.div || 'mjpeg';
    self.camera = options.camera || new RFH.ROSCameraModel();
    self.head = options.head || new Pr2Head(self.ros);
    
    var hfov = 1;
    var vfov = 0.75; //FOV of kinect is ~1 radians wide, 0.75 radians tall
    var SCALE = 0.8; //Scale large motions so we don't over shoot
    var lookAreas = $('.map-look');
    for (var i = 0; i < lookAreas.length; i += 1) {
        var dx = 0, dy = 0;
        var classes = lookAreas[i].classList;
        if (classes.contains("top")) { dy = -SCALE  * vfov };
        if (classes.contains("bottom")) { dy = SCALE  * vfov };
        if (classes.contains("left")) { dx = SCALE * hfov };
        if (classes.contains("right")) { dx = -SCALE * hfov };
        $(lookAreas[i]).on('click.rfh-look', {dx: dx, dy: dy}, function (event) {
            self.head.delPosition(event.data.dx, event.data.dy); 
            event.stopPropagation();
        } );
    }

    self.pointHead = function (e) {
        var pt = RFH.positionInElement(e); 
        var px = (pt[0]/e.target.clientWidth) * self.camera.width;
        var py = (pt[1]/e.target.clientHeight) * self.camera.height;
        var xyz =  self.camera.projectPixel(px, py);
        self.head.pointHead(xyz[0], xyz[1], xyz[2], self.camera.frame_id);
    };

    self.start = function () {
        $('#'+self.div + '-image').addClass("cursor-eyes").on("click.rfh-look", self.pointHead);
        $('.map-look').css("display","block");
        console.log('Looking task started');
    };

    self.stop = function () {
        $('#'+self.div + '-image').removeClass("cursor-eyes").off("click.rfh-look");
        $('.map-look').css("display","none");
        console.log('Looking task stopped');
    };
}

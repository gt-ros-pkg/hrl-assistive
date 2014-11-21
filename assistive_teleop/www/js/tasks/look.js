RFH.Look = function (options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.div = options.div || 'markers';
    self.camera = options.camera || new RFH.ROSCameraModel();
    self.head = options.head || new Pr2Head(self.ros);
    self.buttonText = 'Look';
    self.buttonClass = 'look-button';
    self.thresholds = options.thresholds || {top:0.15, bottom: 0.15,
                                             right: 0.15, left: 0.15};

    self.start = function () {
        self.refreshAreas();
        $('#'+self.div).addClass("cursor-eyes").on("click.rfh", self.pointHead);
        $('.map-look').css("display","block");
    }
    
    self.stop = function () {
        $('#'+self.div).removeClass("cursor-eyes").off("click.rfh");
        $('.map-look').css("display","none").off('click.rfh');
    };

    self.refreshAreas = function () {
        var width = $('#'+self.div).width();
        var height = $('#'+self.div).height();
        var dx = 0, dy = 0;
        var hfov = 1, vfov = 0.75; //FOV of kinect is ~1 radians wide, 0.75 radians tall
        var SCALE = 0.8; //Scale large motions so we don't over shoot
        var lookAreas = $('.map-look');

        for (var i = 0; i < lookAreas.length; i += 1) {
            var newCSS = {};
            if (lookAreas[i].classList.contains("top")) {
                newCSS["top"] = "0px";
                newCSS["height"] = (self.thresholds.top * height) + "px";
                dy = -SCALE  * vfov;
            } else if (lookAreas[i].classList.contains("bottom")) {
                newCSS["bottom"] = "0px";
                newCSS["height"] = (self.thresholds.bottom * height) + "px";
                dy = SCALE  * vfov;
            } else {
                newCSS["top"] = (self.thresholds.top * height) + "px";
                newCSS["height"] = height - ((self.thresholds.bottom + self.thresholds.top) * height) + "px";
                dy = 0;
            }

            if (lookAreas[i].classList.contains("left")) {
                newCSS["left"] = "0px";
                newCSS["width"] = (self.thresholds.left * width) + "px";
                dx = SCALE * hfov;
            } else if (lookAreas[i].classList.contains("right")) {
                newCSS["right"] = "0px";
                newCSS["width"] = (self.thresholds.right * width) + "px";
                dx = -SCALE * hfov;
            } else {
                newCSS["left"] = (self.thresholds.left * width) + "px";
                newCSS["width"] = width - ((self.thresholds.left + self.thresholds.right) * width) + "px";
                dx = 0;
            }

            $(lookAreas[i]).css(newCSS).on('click.rfh', {dx: dx, dy: dy}, function (event) {
                    self.head.delPosition(event.data.dx, event.data.dy); 
                    event.stopPropagation();
                    } );
        }
    }

    self.pointHead = function (e) {
        var pt = RFH.positionInElement(e); 
        var px = (pt[0]/e.target.clientWidth) * self.camera.width;
        var py = (pt[1]/e.target.clientHeight) * self.camera.height;
        var xyz =  self.camera.projectPixel(px, py);
        self.head.pointHead(xyz[0], xyz[1], xyz[2], self.camera.frame_id);
    }
}

assistive_teleop.Look = function (options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.div = options.div || 'markers';
    self.camera = options.camera || new assistive_teleop.ROSCameraModel();
    self.head = options.head || new Pr2Head(self.ros);

    self.thresholds = options.thresholds || {top:0.15,
                                             bottom: 0.85,
                                             right: 0.85,
                                             left: 0.15};
    self.start = function () {
        $('#'+self.div+' canvas').on('mousemove.rfh', self.setCursor);
    }

    self.getRegion = function (e) {
        var pt = assistive_teleop.positionInElement(e) 
        console.log("Target Element Size: (x="+e.target.clientWidth+", y="+e.target.clientHeight+")");
        var pct_x = pt[0]/e.target.clientWidth;
        var pct_y = pt[1]/e.target.clientHeight;
        var l = false, r = false, t = false, b = false;
        if (pct_x < self.thresholds.left) {l = true};
        if (pct_x > self.thresholds.right) {r = true};
        if (pct_y > self.thresholds.bottom) {b = true};
        if (pct_y < self.thresholds.top) {t = true};
        //Divide rectangular region into top/bottom/left/right/center combos
        //Numbers correspond to layout on keyboard numpad;
        if (!(b || t || l || r)) return 5; //largest area -> most common -> check first
        if (b) {
            if (l) return 1;
            else if (r) return 3;
            else return 2;
            }
        if (t) {
            if (l) return 7;
            else if (r) return 9;
            else return 8;
            }
        if (l) return 4;
        else return 6;
    }

    self.onClick = function (e) {
        var dx = 0, dy = 0, point = false;
        var hfov = 1, vfov = 0.75; //FOV of kinect is ~1 radians wide, 0.75 radians tall
        var SCALE = 0.8; //Scale large motions so we don't over shoot
        switch (self.getRegion(e)) {
            case 1: 
                var dx = SCALE * hfov;
                var dy = SCALE * vfov;
                break;
            case 2: 
                var dy = SCALE * vfov;
                break;
            case 3: 
                var dx = -SCALE * hfov;
                var dy = SCALE * vfov;
                break;
            case 4: 
                var dx = SCALE * hfov;
                break;
            case 5: 
                point = true;
                break;
            case 6: 
                var dx = -SCALE * hfov;
                break;
            case 7: 
                var dx = SCALE * hfov;
                var dy = -SCALE * vfov;
                break;
            case 8: 
                var dy = -SCALE * vfov;
                break;
            case 9: 
                var dx = -SCALE * hfov;
                var dy = -SCALE * vfov;
                break;
        }
        if (point) {
            var pt = assistive_teleop.positionInElement(e); 
            var px = (pt[0]/e.target.clientWidth) * self.camera.width;
            var py = (pt[1]/e.target.clientHeight) * self.camera.height;
            var xyz =  self.camera.projectPixel(px, py);
            self.head.pointHead(xyz[0], xyz[1], xyz[2], self.camera.frame_id);
        } else {
            self.head.delPosition(dx,dy);
        }
        
    }

    self.setCursor = function (e) {
//        switch (self.getRegion(pct_x, pct_y)) {
//            case 1: var dir = 'down-left';
//                    break;
//            case 2: var dir = 'down';
//                    break;
//            case 3: var dir = 'down-right';
//                    break;
//            case 4: var dir = 'left';
//                    break;
//            case 5: var dir = '';
//                    break;
//            case 6: var dir = 'right';
//                    break;
//            case 7: var dir = 'up-left';
//                    break;
//            case 8: var dir = 'up';
//                    break;
//            case 9: var dir = 'up-right';
//            }
//        $('#'+self.div).css({'cursor': 'url("css/cursors/eyes/eyes-'+dir+'.png")'});
        var region = self.getRegion(e);
        switch (region) {
            case 1: var dir = 'sw-resize';
                    break;
            case 2: var dir = 's-resize';
                    break;
            case 3: var dir = 'se-resize';
                    break;
            case 4: var dir = 'w-resize';
                    break;
            case 5: var dir = 'crosshair';
                    break;
            case 6: var dir = 'e-resize';
                    break;
            case 7: var dir = 'nw-resize';
                    break;
            case 8: var dir = 'n-resize';
                    break;
            case 9: var dir = 'ne-resize';
            }
        $('#'+self.div).css({'cursor': dir});
    }
}


var initLook = function ()  {
    assistive_teleop.tasks['look'] = new assistive_teleop.Look({ros: assistive_teleop.ros, 
                                                             div: 'markers',
                                                             camera: assistive_teleop.mjpeg.cameraModel});
    $('#menu-item-look').button().on('click.rfh', function(){assistive_teleop.tasks.look.start()});
    $('#clickable-canvas').on('click.rfh', assistive_teleop.tasks.look.onClick);
}

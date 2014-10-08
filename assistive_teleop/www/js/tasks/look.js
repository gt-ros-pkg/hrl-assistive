RFH.Look = function (options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.div = options.div || 'markers';
    self.camera = options.camera || new RFH.ROSCameraModel();
    self.head = options.head || new Pr2Head(self.ros);
    self.buttonText = 'Looking';
    self.buttonClass = 'look-button';

    self.thresholds = options.thresholds || {top:0.15,
                                             bottom: 0.85,
                                             right: 0.85,
                                             left: 0.15};
    self.cursorClasses = {1: 'cursor-eyes-down-left',
                          2: 'cursor-eyes-down',
                          3: 'cursor-eyes-down-right',
                          4: 'cursor-eyes-left',
                          5: 'cursor-eyes',
                          6: 'cursor-eyes-right',
                          7: 'cursor-eyes-up-left',
                          8: 'cursor-eyes-up',
                          9: 'cursor-eyes-up-right'}

    self.start = function () {
        $('#'+self.div+' canvas').on('mousemove.rfh', self.setCursor);
    }
    
    self.stop = function () {
        $('#'+self.div+' canvas').off('mousemove.rfh');
        for ( var idx in self.cursorClasses ) {
            $('#'+self.div+' canvas').removeClass( self.cursorClasses[ idx ] );
        }
    };

    self.setCursor = function (e) {
        for ( var idx in self.cursorClasses ) {
            $('#'+self.div+' canvas').removeClass( self.cursorClasses[ idx ] );
        }
        $('#'+self.div+' canvas').addClass(self.cursorClasses[self.getRegion(e)]);
    };

    self.getRegion = function (e) {
        var pt = RFH.positionInElement(e) 
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
            var pt = RFH.positionInElement(e); 
            var px = (pt[0]/e.target.clientWidth) * self.camera.width;
            var py = (pt[1]/e.target.clientHeight) * self.camera.height;
            var xyz =  self.camera.projectPixel(px, py);
            self.head.pointHead(xyz[0], xyz[1], xyz[2], self.camera.frame_id);
        } else {
            self.head.delPosition(dx,dy);
        }
    }
}

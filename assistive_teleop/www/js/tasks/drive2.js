RFH.Drive = function (options) {
    "use strict";
    var self = this;
    var options = options || {};
    self.name = options.name || 'drivingTask';
    self.ros = options.ros;
    self.div = options.targetDiv || 'markers';
    self.head = options.head;
    self.camera = options.camera;
    self.base = options.base;
    self.buttonText = 'Drive';
    self.buttonClass = 'drive-button';
    self.timer = null;
    self.lines = {'left':null,
                  'center':null,
                  'right':null};
    self.lineCenterOffset = 0; //5% image width offset between lines
    self.lineWidthOffset = 0; //5% image width offset between lines
    self.clamp = function (x,a,b) {
        return ( x < a ) ? a : ( ( x > b ) ? b : x );
    }
    self.sign = function (x) { 
        return typeof x === 'number' ? x ? x < 0 ? -1 : 1 : x === x ? 0 : NaN : NaN;
    }
   
    self.driveSVG = Snap('#drive-lines');

    self.getLineParams = function (event) {
        var target = RFH.positionInElement(event); 
        var originX = $('#drive-lines').width()/2 + self.lineCenterOffset;
        var originY = $('#drive-lines').height();
        var tx = -(target[0]-originX);
        var ty = -(target[1]-originY);
        var ang = Math.atan2(tx, ty);
        var mag = Math.sqrt(tx*tx+ty*ty);
        var pathRad = 0.5*Math.abs(mag/Math.sin(ang));
        var cx = originX + ((ang >= 0 ) ? -pathRad : pathRad) ;
        return {'cx':cx, 'originX':originX, 'originY':originY, 'rad':pathRad};
    };

    self.createLinesCB = function (event) {
        var lp = self.getLineParams(event); 
        if (lp.rad === Infinity) {
            lp.rad = 10000
            lp.cx = lp.rad + lp.originX;
            }
        self.lines['center'] = self.driveSVG.circle(lp.cx, lp.originY, lp.rad).attr({
                                    "id":"dl-circle-center",
                                    "stroke-width": 10,
                                    "stroke-opacity":0.6,
                                    "fill":"none"}).transform('s1,0.84,0,'+lp.originY.toString());
        self.lines['left'] = self.lines['center'].clone().attr({
                                    'id':'dl-circle-left',
                                    'rad':(lp.rad - self.lineWidthOffset),
                                    'cx':lp.cx});
        self.lines['right'] = self.lines['center'].clone().attr({
                                    'id':'dl-circle-right',
                                    'rad':(lp.rad + self.lineWidthOffset),
                                    'cx':lp.cx});
    };
    self.driveSVG.node.onmouseenter = self.createLinesCB;

    self.removeLinesCB = function (event) {
        self.lines['center'].remove();
        self.lines['left'].remove();
        self.lines['right'].remove();
        self.lines = {};
    };
    self.driveSVG.node.onmouseleave = self.removeLinesCB;

    self.updateLinesCB = function (event) {
        var lp = self.getLineParams(event);
        if (lp.rad === Infinity) { return }
        self.lines['center'].attr({'cx':lp.cx, 'cy':lp.originY, 'r': lp.rad});
        self.lines['left'].attr({'cx':lp.cx, 'cy':lp.originY, 'r': Math.max((lp.rad-self.lineWidthOffset), 0)});
        self.lines['right'].attr({'cx':lp.cx, 'cy':lp.originY, 'r': Math.max((lp.rad+self.lineWidthOffset), 0)});
    };
//    self.driveSVG.node.onmousemove = self.updateLinesCB;

 
    self.getWorldPath = function (event) {
        var rtxy = self.getRTheta(event); //Get real-world point in base frame
        var qx = rtxy[2];
        var qy = rtxy[3];
        //Find Center of circle (tangent to base center, through clicked point) in world
        // Px,Py = (0,0) (base frame origin)
        // Cy = Py = 0 (forward motion is tangent to circle)
        // Distance from Center to P, Q (clicked point) must be equal
        var Cy = (qx*qx + qy*qy) / (2*qy);
        var Cx = 0;
        var Rworld = Math.abs(Cy);
        console.log(Cx, Cy, Rworld);
        // Solve for circle matrix parameters.
        // Ref: http://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
        // For circle: B = 0, A = C;
        // A(Px*Px+Py*Py) + DPx + EPy + F = 0 (Px=Py=0 --> F=0) Solve remaining 3 params:
        // A(qx*qz+qy*qy) + Dqx + Eqy (+F=0) = 0
        // -0.5D = Cx (from equation for center points, reduced, with B/A/C subs above).
        // -0.5E = Cy
        var D = -2*Cx;
        var E = -2*Cy;
        var A = -(D*qx + E*qy) / (qx*qx + qy*qy)
        var circleMat = [[A,   0,   D/2],
                         [0,   A,   E/2],
                         [D/2, E/2, 0]];

        
                        


    }
    self.driveSVG.node.onclick = self.getWorldPath;

    self.updateLineOffsets = function (event) {
        var width =$('#drive-lines').width();
        self.lineCenterOffset = 0;//-0.08*width;
        self.lineWidthOffset = 0.18*width;
    };

    self.headStops = ['back-left', 'left','forward','right','back-right'];
    self.headStopAngles = {'back-right':[-2.85, 1.35],
                           'right':[-Math.PI/2, 1.35],
                           'forward':[0.0, 1.0],
                           'left':[Math.PI/2, 1.35],
                           'back-left': [2.85, 1.35]}

    self.baseCenters = {'back-right':[0.444, 1.103], //From 1-time test
                        'right':[0.436, 1.065],
                         'forward':[0.459, 1.203],
                         'left':[0.473, 1.086],
                         'back-left': [0.455, 1.108]}

    self.edges = {'front': 0.334,
                  'back': -0.334,
                  'left': 0.334,
                  'right': -0.334}

    self.onLeft = function (x,y) {
        return y > self.edge['left'] ? true : false;
    }

    self.toLeft = function (e) {
        var newStop = self.headStops[self.headStops.indexOf(self.currentStop) - 1];
        if (newStop) {
            self.moveToStop(newStop);
        }
    }

    self.toRight = function (e) {
        var newStop = self.headStops[self.headStops.indexOf(self.currentStop) + 1];
        if (newStop) {
            self.moveToStop(newStop);
        }
    }

    $('.drive-look.left').on('click.rfh', self.toLeft)
    $('.drive-look.right').on('click.rfh', self.toRight)

    self.getNearestStop = function () {
        var currentPan = self.head.state[0];
        var nearestStop = 'forward'; //Good default assumption;
        var del = 2*Math.PI; //Initialize too high;
        for (var i=0; i < self.headStops.length; i++ ) {
            var stop = self.headStops[i];
            var dist = Math.abs(self.headStopAngles[stop][0] - currentPan)
            if (dist <= del) {
                del = dist;
                nearestStop = stop;
            }
        }
        return nearestStop;
    }

    self.moveToStop = function (stopName) {
        var angs = self.headStopAngles[stopName];
        self.currentStop = stopName;
        self.head.setPosition(angs[0], angs[1]);
    }

    self.start = function () {
        // everything i can think of to not get stuck driving...
        $(document).on("mouseleave.rfh mouseout.rfh", self.setUnsafe);
        $('#drive-lines').on('mouseleave.rfh mouseout.rfh', self.setUnsafe)
//        $('#drive-lines').on('mousedown.rfh', self.driveGo);
        $('#drive-lines').on('mouseup.rfh', self.driveStop);
        $('#drive-lines').on('blur.rfh', self.driveStop);
        $('.drive-ctrl').show();
        self.moveToStop(self.getNearestStop());
        self.updateLineOffsets();
        $('#drive-lines').on('resize.rfh', self.updateLineOffsets)
    }

    self.stop = function () {
        $(document).off("mouseleave.rfh mouseout.rfh");
        $('#'+self.div).removeClass('drive-safe');
        $('#'+self.div).off('mouseleave.rfh mouseout.rfh mousedown.rfh mouseup.rfh hover')
        $('.drive-ctrl').hide();
    }

    self.driveGo = function (event) {
        if (event.which === 1) { //Only react to left mouse button
            self.setSafe();
            $('#'+self.div).on('mousemove.rfh', self.driveToGoal); 
            self.driveToGoal(event);
        } else {
            self.driveStop();
        }
    }

    self.driveStop = function (event) {
        self.setUnsafe();
        $('#'+self.div).off('mousemove.rfh');
    } 

    self.setSafe = function () {
        $('#'+self.div).addClass('drive-safe');
    }

    self.setUnsafe = function (event) {
        //alert("Unsafe: "+event.type);
        clearTimeout(self.timer);
        $('#'+self.div).removeClass('drive-safe');
    }

    self.driveToGoal = function (event) {
        clearTimeout(self.timer);
        try { 
            var rtxy = self.getRTheta(event);
        } catch (err) {
            console.warn(err.message);
            return;
        };
        self.timer = setTimeout(function(){self.sendCmd(rtxy[0], rtxy[1], rtxy[2], rtxy[3]);}, 1);
    }

    self.getRTheta = function (e) {
        var pt = RFH.positionInElement(e); 
        var px = (pt[0]/e.target.clientWidth) * self.camera.width;
        var py = (pt[1]/e.target.clientHeight) * self.camera.height;
        if (self.camera.frame_id === '') {
            alert("Camera position not up to date.  Cannot drive safely.");
            self.camera.updateCameraInfo();
            }
        var xyz = self.camera.projectPixel(px, py, 1.0);
        var pose = new ROSLIB.Pose({position:{x: xyz[0],
                                              y: xyz[1], 
                                              z: xyz[2]}});
        pose.applyTransform(self.camera.transform);
        if (pose.position.z >= self.camera.transform.translation.z) {
            RFH.log('Please click on the ground near the robot to drive.');
            throw new Error("Clicked point not on the ground");
        }
        var z0 = self.camera.transform.translation.z;
        var z1 = pose.position.z;
        var dist = (z0+0.05)/(z0-z1) // -0.05 = z0 - ((z0-z1)/1)*x -> lenght of line to intersection
        var gnd_pt = [0,0,0];
        gnd_pt[0] = self.camera.transform.translation.x + (pose.position.x - self.camera.transform.translation.x) * dist;
        gnd_pt[1] = self.camera.transform.translation.y + (pose.position.y - self.camera.transform.translation.y) * dist; 
        var r = Math.sqrt(gnd_pt[0]*gnd_pt[0] + gnd_pt[1]*gnd_pt[1]);
        var theta = Math.atan2(gnd_pt[1], gnd_pt[0]);
        console.log("R: "+r+", Theta: "+theta);
        return [r, theta, gnd_pt[0], gnd_pt[1]];
    }

    self.sendCmd = function (r, theta, x, y) {
        if (!$('#'+self.div).hasClass('drive-safe')) { //Check for safety
            return;
        }
        var cmd_y = 0, cmd_x = 0, cmd_theta = 0;
        if ( r < 0.6 ) { //Goal close to base, holonomic move in straight line.
           var cmd_x = self.sign(x) * self.clamp(0.1*Math.abs(x), 0.0, 0.2);
           var cmd_y = self.sign(y) * self.clamp(0.1*Math.abs(y), 0.0, 0.2);
        } else {
            cmd_theta = 0.5 * self.clamp( theta, -Math.PI/6, Math.PI/6 );
            if ( theta < Math.PI/6 && theta > -Math.PI/6) {
                //Moving mostly forward, SERVO
               // var cmd_x = self.sign(x) * self.clamp(0.1*Math.abs(x), 0.0, 0.2);
               //var cmd_y = self.sign(y) * self.clamp(0.1*Math.abs(y), 0.0, 0.2);
                var cmd_x = self.clamp(0.1 * x, -0.2, 0.2);
                var cmd_y = self.clamp(0.1 * y, -0.2, 0.2);
            }
        }
        //cmd_x = Math.abs(cmd_x) < 0.02 ? 0 : cmd_x;
        //cmd_y = Math.abs(cmd_y) < 0.02 ? 0 : cmd_y;
        //cmd_theta = Math.abs(cmd_theta) < 0.075 ? 0 : cmd_theta;
        self.base.pubCmd(cmd_x, cmd_y, cmd_theta);
        self.timer = setTimeout(function(){self.sendCmd(r, theta, x, y);}, 50);
    }
}

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
    self.clamp = function (x,a,b) {
        return ( x < a ) ? a : ( ( x > b ) ? b : x );
    }
    self.sign = function (x) { 
        return typeof x === 'number' ? x ? x < 0 ? -1 : 1 : x === x ? 0 : NaN : NaN;
    }
    
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
        $('#'+self.div).on('mouseleave.rfh mouseout.rfh', self.setUnsafe)
        $('#'+self.div).on('mousedown.rfh', self.driveGo);
        $('#'+self.div).on('mouseup.rfh', self.driveStop);
        $('#'+self.div).on('blur.rfh', self.driveStop);
        $('.drive-ctrl').show();
        self.moveToStop(self.getNearestStop());
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

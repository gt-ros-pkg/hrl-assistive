RFH.Drive = function (options) {
    "use strict";
    var self = this;
    var options = options || {};
    self.ros = options.ros;
    self.div = options.div || 'markers';
    self.tfClient = options.tfClient;
    self.camera = options.camera;
    self.base = options.base;
    self.buttonText = 'Drive';
    self.buttonClass = 'drive-button';
    self.headTF = new ROSLIB.Transform();
    self.clamp = function (x,a,b) {
        return ( x < a ) ? a : ( ( x > b ) ? b : x );
    }
    self.sign = function (x) { 
        return typeof x === 'number' ? x ? x < 0 ? -1 : 1 : x === x ? 0 : NaN : NaN;
    }
    
    self.updateHead = function (transform) { self.headTF = transform; }
    self.tryTFSubscribe = function () {
        if (self.camera.frame_id !== '') {
            self.tfClient.subscribe(self.camera.frame_id, self.updateHead);
            console.log("Got camera data, subscribing to TF Frame: "+self.camera.frame_id);
        } else {
            console.log("No camera data -> no TF Transform");
            setTimeout(self.tryTFSubscribe, 500);
            }
    }
    self.tryTFSubscribe();

    self.start = function () {
        //TODO: set informative cursor
        // everything i can think of to not get stuck driving...
        $(document).on("mouseleave mouseout", self.setunsafe);
        $('#'+self.div+' canvas').on('mouseleave mouseout', self.setunsafe)
        $('#'+self.div+' canvas').on('mousedown.rfh', self.drive_go);
        $('#'+self.div+' canvas').on('mouseup.rfh', self.drive_stop);
        $('#'+self.div+' canvas').hover( self.setsafe, self.setunsafe );
    }

    self.stop = function () {
        $(document).off("mouseleave mouseout");
        $('#'+self.div+' canvas').removeClass('drive-safe');
        $('#'+self.div+' canvas').off('mouseleave mouseout mousedown.rfh mouseup.rfh hover')
    }

    self.drive_go = function (event) {
        $('#'+self.div+' canvas').on('mousemove', self.driveToGoal); 
    }

    self.drive_stop = function (event) {
        self.setUnsafe();
        $('#'+self.div+' canvas').off('mousemove');
        self.setSafe();
   } 

    self.setSafe = function () {
        $('#'+self.div+' canvas').addClass('drive-safe');
    }

    self.setUnsafe = function (event) {
        $('#'+self.div+' canvas').removeClass('drive-safe');
    }

    self.driveToGoal = function (event) {
        try { 
            var rtxy = self.getRTheta(event);
        } catch (err) {
            return;
        };
        self.sendCmd(rtxy[0], rtxy[1], rtxy[2], rtxy[3]); 
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
        pose.applyTransform(self.headTF);
        if (pose.position.z >= self.headTF.translation.z) {
            RFH.log('Please click on the ground near the robot to drive.');
            throw new Error("Clicked point not on the ground");
        }
        var z0 = self.headTF.translation.z;
        var z1 = pose.position.z;
        var dist = (z0+0.05)/(z0-z1) // -0.05 = z0 - ((z0-z1)/1)*x -> lenght of line to intersection
        var gnd_pt = [0,0,0];
        gnd_pt[0] = self.headTF.translation.x + (pose.position.x - self.headTF.translation.x) * dist;
        gnd_pt[1] = self.headTF.translation.y + (pose.position.y - self.headTF.translation.y) * dist; 
        var r = Math.sqrt(gnd_pt[0]*gnd_pt[0] + gnd_pt[1]*gnd_pt[1]);
        var theta = Math.atan2(gnd_pt[1], gnd_pt[0]);
        console.log("R: "+r+", Theta: "+theta);
        return [r, theta, gnd_pt[0], gnd_pt[1]];
    }

    self.sendCmd = function (r, theta, x, y) {
        if (!$('#'+self.div+' canvas').hasClass('drive-safe')) { //Check for safety
            return;
        }
        var cmd_y = 0, cmd_x = 0, cmd_theta = 0;
        if ( r < 0.6 ) { //Goal close to base, holonomic move in straight line.
           var cmd_x = self.sign(x) * self.clamp(0.1*Math.abs(x), 0.0, 0.2);
           var cmd_y = self.sign(y) * self.clamp(0.1*Math.abs(y), 0.0, 0.2);
        } else {
            cmd_theta = 0.5 * self.clamp( theta, -Math.PI/6, Math.PI/6 );
            if ( theta < Math.PI/6 && theta < -Math.PI/6) {
                //Moving mostly forward, SERVO
                var cmd_x = self.sign(x) * self.clamp(0.1*Math.abs(x), 0.0, 0.2);
                var cmd_y = self.sign(y) * self.clamp(0.1*Math.abs(y), 0.0, 0.2);
            }
        }
        self.base.pubCmd(cmd_x, cmd_y, cmd_theta);
        setTimeout(function(){self.sendCmd(r, theta, x, y);}, 100);
    }
}

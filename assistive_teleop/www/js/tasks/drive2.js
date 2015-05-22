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
    self.baseOffset = [0.0, 0.35]; //Half-width of PR2 Base [x - front/back, y-left/right]
    self.Ndots = 25;
    self.lineWidthOffset = 0.18; //5% image width offset between lines
    self.clamp = function (x,a,b) {
        return ( x < a ) ? a : ( ( x > b ) ? b : x );
    }
    self.sign = function (x) { 
        return typeof x === 'number' ? x ? x < 0 ? -1 : 1 : x === x ? 0 : NaN : NaN;
    }
   
    self.driveSVG = Snap('#drive-lines');
    self.initPathMarkers = function (Ndots, d) {
        var opacity = numeric.linspace(1, 0.05, Ndots);
        self.lines['left'] = self.driveSVG.g();
        self.lines['center'] = self.driveSVG.g();
        self.lines['right'] = self.driveSVG.g();
        for (var i=0; i<Ndots; i += 1) {
            var c = self.driveSVG.paper.circle(0, 0, d);
            c.attr({'fill-opacity':opacity[i]});
            self.lines['left'].add(c);
            self.lines['right'].add(c.clone());
            self.lines['center'].add(c.clone());
        }
        self.lines['left'].attr({'fill':'rgb(0,188,212)'});
        self.lines['right'].attr({'fill':'rgb(0,188,212)'});
        self.lines['center'].attr({'fill':'rgb(33,150,243)'});
    };
    self.initPathMarkers(self.Ndots, 4.5);

    self.showLinesCB = function (event) {
        self.lines['left'].attr({'display':'block'});
        self.lines['right'].attr({'display':'block'});
        self.lines['center'].attr({'display':'block'});
    };
    $(self.driveSVG.node).on('mouseenter.rfh', self.showLinesCB);

    self.removeLinesCB = function (event) {
        if ($(event.relatedTarget).hasClass('turn-signal')){return};
        self.lines['left'].attr({'display':'none'});
        self.lines['right'].attr({'display':'none'});
        self.lines['center'].attr({'display':'none'});
    };
    $(self.driveSVG.node).on('mouseleave.rfh', self.removeLinesCB);

    self.getWorldPath = function (rtxy) {
        var T = rtxy.slice(2); //Target (clicked) Point on floor in real world
        var B = [0, 0]; //origin of motion (base movement moves center of robot)
        var A = [0, 0]; //point straight ahead.
        var BL = [0, 0]; //Left base line point
        var BR = [0, 0]; //Right base line point
        var C = [0, 0];
        var side = '';
        var dirAngleOffset;
        switch (self.currentStop) {
            case 'forward':
                A = [B[0] + 1, B[1]];
                BL = [B[0] + self.baseOffset[0], B[1] + self.baseOffset[1]] //Left and right guide-line reference points
                BR = [B[0] + self.baseOffset[0], B[1] - self.baseOffset[1]] 
                C[0] = B[0]; // Center is on line with base center
                //Center is equidistant from base center to target -- Must for corner of a square with base and target.
                C[1] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[0]*(B[0] - T[0]) ) / (2 * (B[1] - T[1]));
                side = (T[1] > B[1]) ? "left" : "right";
                dirAngleOffset = 0;
                break;
            case 'back-left':
            case 'back-right':
                A = [B[0] - 1, B[1]];
                BL = [B[0] - self.baseOffset[0], B[1] - self.baseOffset[1]]
                BR = [B[0] - self.baseOffset[0], B[1] + self.baseOffset[1]]
                C[0] = B[0]; // Center is on line with base center
                C[1] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[0]*(B[0] - T[0]) ) / (2 * (B[1] - T[1]));
                side = (T[1] < B[1]) ? "left" : "right";
                dirAngleOffset = Math.PI;
                break;
            case 'left':
                A = [B[0], B[1] + 1];
                BL = [B[0] - self.baseOffset[1], B[1] + self.baseOffset[0]]
                BR = [B[0] + self.baseOffset[1], B[1] + self.baseOffset[0]]
                C[1] = B[1];
                C[0] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[1]*(B[1] - T[1]) ) / (2 * (B[0] - T[0]));
                side = (T[0] < B[0]) ? "left" : "right";
                dirAngleOffset = Math.PI/2;;
                break;
            case 'right':
                A = [B[0], B[1] - 1];
                BL = [B[0] + self.baseOffset[1], B[1] - self.baseOffset[0]]
                BR = [B[0] - self.baseOffset[1], B[1] - self.baseOffset[0]]
                C[1] = B[1];
                C[0] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[1]*(B[1] - T[1]) ) / (2 * (B[0] - T[0])); 
                side = (T[0] > B[0]) ? "left" : "right";
                dirAngleOffset = -Math.PI/2;
                break;
        }
        var R = Math.sqrt( Math.pow(C[0]-B[0], 2) + Math.pow(C[1]-B[1], 2) )
        var RL = Math.sqrt( Math.pow(C[0]-BL[0], 2) + Math.pow(C[1]-BL[1], 2) )
        var RR = Math.sqrt( Math.pow(C[0]-BR[0], 2) + Math.pow(C[1]-BR[1], 2) )
        var CB = numeric.sub(C,B);
        var CT = numeric.sub(C,T);
        var theta = Math.acos(numeric.dot(CB, CT)/(R*R));
        return {'R':R, 'RL':RL, 'RR':RR, 'side':side, 'dirAngleOffset': dirAngleOffset, 'C':C, 'B':B, 'T':T, 'theta':theta};
        }

    self.drawPath = function (rtxy) {
        // Find points around the circle from start to target location.
        var path = self.getWorldPath(rtxy);
        self.cmd = self.calcCmd(path);
        path.theta *= 1.4; // Extend path past point targeted by mouse
        var Ndots = 25;
        var minAng = (path.side === 'right') ? Math.PI/2 + path.dirAngleOffset : -Math.PI/2 + path.dirAngleOffset;
        var maxAng = (path.side === 'right') ? minAng - path.theta : minAng + path.theta;
        var angs = numeric.linspace(minAng, maxAng, Ndots);
        var opacity = numeric.linspace(1, 0.05, Ndots);
        var w = $(self.driveSVG.node).width();
        var h = $(self.driveSVG.node).height();
        var pts = [];

        for (var i in angs) {
            pts.push([path.C[0] + path.R*Math.cos(angs[i]), path.C[1] + path.R*Math.sin(angs[i]), 0]);
        }
        var imgpts = self.camera.projectPoints(pts, 'base_link');
        for (var i=0; i<imgpts.length; i += 1) {
            imgpts[i][0] *= w;
            imgpts[i][1] *= h;
            self.lines['center'][i].attr({'cx':imgpts[i][0], 'cy':imgpts[i][1]});
        }

        if (path.side === 'right' || path.RL < path.R) {
            pts = [];
            for (var i in angs) {
                pts.push([path.C[0] + path.RL*Math.cos(angs[i]), path.C[1] + path.RL*Math.sin(angs[i]), 0]);
            }
            var imgpts = self.camera.projectPoints(pts, 'base_link');
            //Draw points
            for (var i=0; i<imgpts.length; i += 1) {
                imgpts[i][0] *= w;
                imgpts[i][1] *= h;
                self.lines['left'][i].attr({'cx':imgpts[i][0], 'cy':imgpts[i][1]});
            }
        }

        if (path.side == 'left' ||  path.RR < path.R) {
            pts = [];
            for (var i in angs) {
                pts.push([path.C[0] + path.RR*Math.cos(angs[i]), path.C[1] + path.RR*Math.sin(angs[i]), 0]);
            }
            var imgpts = self.camera.projectPoints(pts, 'base_link');
            //Draw points
            for (var i=0; i<imgpts.length; i += 1) {
                imgpts[i][0] *= w;
                imgpts[i][1] *= h;
                self.lines['right'][i].attr({'cx':imgpts[i][0], 'cy':imgpts[i][1]});
            }
        }
    };

    self.calcCmd = function (path) {
        var linV = self.clamp(0.2*path.R*path.theta, 0.0, 0.4);
        var cmd_theta = Math.tan(linV/path.R);
        cmd_theta *= (path.side === 'left') ? 1 : -1;
        var cmd_x = 0;
        var cmd_y = 0;
        switch (self.currentStop) {
            case "forward":
                cmd_x = linV;
                break;
            case "back-left":
            case "back-right":
                cmd_x = -linV;
                break;
            case "left":
                cmd_y = linV;
                break;
            case "right":
                cmd_y = -linV;
                break;
        }
        return {'x':cmd_x, 'y':cmd_y, 'theta':cmd_theta};
    }

    self.drawSlidePath = function (rtxy) {
        switch (self.currentStop) {
            case "forward":
                var B = [0.2, 0, 0];
                var L = [0.33*B[0]+self.baseOffset[0], B[1] + self.baseOffset[1], 0];
                var R = [0.33*B[0]+self.baseOffset[0], B[1] - self.baseOffset[1], 0];
                break;
            case "back-left":
            case "back-right":
                var B = [-0.2, 0, 0];
                var L = [-0.33*B[0]+self.baseOffset[0], B[1] + self.baseOffset[1], 0];
                var R = [-0.33*B[0]+self.baseOffset[0], B[1] - self.baseOffset[1], 0];
                break;
            case "left":
                var B = [0, 0.2, 0];
                var L = [B[0]-self.baseOffset[1], 0.5*B[1] + self.baseOffset[0], 0];
                var R = [B[0]+self.baseOffset[1], 0.5*B[1] + self.baseOffset[0], 0];
                break;
            case "right":
                var B = [0, -0.2, 0];
                var L = [B[0]+self.baseOffset[1], 0.5*B[1] - self.baseOffset[0], 0];
                var R = [B[0]-self.baseOffset[1], 0.5*B[1] - self.baseOffset[0], 0];
                break;
        }
        var T = rtxy.slice(2); //Target (clicked) Point on floor in real world
        var pts = [B, [T[0], T[1], 0], L, R];
        var imgpts = self.camera.projectPoints(pts, 'base_link');
        var w = $(self.driveSVG.node).width();
        var h = $(self.driveSVG.node).height();
        for (var i=0; i<imgpts.length; i += 1) {
            imgpts[i][0] *= w;
            imgpts[i][1] *= h;
        }
        var slide = [imgpts[1][0] - imgpts[0][0], imgpts[1][1] - imgpts[0][1]];
        var slideX = numeric.linspace(0, slide[0], self.Ndots);
        var slideY = numeric.linspace(0, slide[1], self.Ndots);
        for (var i=0; i<self.Ndots; i += 1) {
            self.lines['left'][i].attr({'cx':imgpts[2][0] + slideX[i], 'cy':imgpts[2][1] + slideY[i]});
            self.lines['center'][i].attr({'cx':imgpts[0][0] + slideX[i], 'cy':imgpts[0][1] + slideY[i]});
            self.lines['right'][i].attr({'cx':imgpts[3][0] + slideX[i], 'cy':imgpts[3][1] + slideY[i]});
        }
        var delX = self.clamp((T[0] - B[0])/2, -0.45, 0.45);
        var delY = self.clamp((T[1] - B[1])/2, -0.45, 0.45);
        self.cmd = {'x':delX, 'y':delY, 'theta':0};
    }

    self.updateVis = function (event) {
        var rtxy = self.getRTheta(event); //Get real-world point in base frame
        if (rtxy[0] < 0.45) {
            self.drawSlidePath(rtxy);
        } else {
            self.drawPath(rtxy);
        }
    }
    $(self.driveSVG.node).on("mousemove.rfh", self.updateVis);

    self.headStops = ['back-left', 'left','forward','right','back-right'];
    self.headStopAngles = {'back-right':[-2.85, 1.35],
                           'right':[-Math.PI/2, 1.35],
                           'forward':[0.0, 1.35],
                           'left':[Math.PI/2, 1.35],
                           'back-left': [2.85, 1.35]}

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

    self.driveDirIcon = Snap("#drive-dir-icon");
    Snap.load('./css/icons/drive-direction-icon.svg', function (icon_svg) {
        self.driveDirIcon.append(icon_svg.select('g'));
       console.log("Drive Direction Icon Loaded"); 
    });

    $('.turn-signal.left').on('mouseenter', function (event) {
        self.cmd = {'x':0, 'y':0, 'theta':0.1*Math.PI};
        self.lines['left'].attr({'display':'block'});
        self.lines['right'].attr({'display': 'block'});
        self.lines['center'].attr({'display':'block'});
        var w = $(self.driveSVG.node).width();
        var h = $(self.driveSVG.node).height();
        var pts = self.camera.projectPoints([[0,0,0],[0,0.2,0]],'base_link');
        var cx = pts[0][0]*w;
        var cy = pts[0][1]*h;
        var r = numeric.norm2(numeric.sub(pts[1], pts[0]))*w;
        self.lines['center'].children().forEach(function(c){c.attr({'cx':cx, 'cy':cy})});
        var angs = numeric.linspace(-Math.PI, 0.85*Math.PI, self.Ndots);
        var Rcircles = self.lines['right'].children();
        var Lcircles = self.lines['left'].children();
        for (var i in angs) {
            Rcircles[i].attr({'cx':cx + 0.5*r*Math.cos(angs[i]),
                              'cy':cy + 0.5*r*Math.sin(angs[i])});
            Lcircles[i].attr({'cx':cx + r*Math.cos(angs[i]),
                              'cy':cy + r*Math.sin(angs[i])});
        }
        self.spinLeft = function () {
            self.lines['left'].attr({'transform':'r0,'+cx+','+cy});
            self.lines['right'].attr({'transform':'r0,'+cx+','+cy});
            self.lines['left'].animate({'transform':'r-360,'+cx+','+cy}, 1500, mina.linear);
            self.lines['right'].animate({'transform':'r-360,'+cx+','+cy}, 1500, mina.linear);
        };
        self.spinLeft();
        self.leftSpinTimer = setInterval(self.spinLeft, 1500);
    }).on('mouseleave', function (event) {
        clearTimeout(self.leftSpinTimer);
        self.lines['left'].stop();
        self.lines['right'].stop();
        self.lines['left'].attr({'transform':'r0', 'display':'none'});
        self.lines['right'].attr({'transform':'r0', 'display':'none'});
        self.lines['center'].attr({'display':'none'});
    });

    $('.turn-signal.right').on('mouseenter', function (event) {
        self.cmd = {'x':0, 'y':0, 'theta':-0.1*Math.PI};
        self.lines['left'].attr({'display':'block'});
        self.lines['right'].attr({'display': 'block'});
        self.lines['center'].attr({'display':'block'});
        var w = $(self.driveSVG.node).width();
        var h = $(self.driveSVG.node).height();
        var pts = self.camera.projectPoints([[0,0,0],[0,0.2,0]],'base_link');
        var cx = pts[0][0]*w;
        var cy = pts[0][1]*h;
        var r = numeric.norm2(numeric.sub(pts[1], pts[0]))*w;
        self.lines['center'].children().forEach(function(c){c.attr({'cx':cx, 'cy':cy})});
        var angs = numeric.linspace(-Math.PI, 0.85*Math.PI, self.Ndots);
        var Rcircles = self.lines['right'].children();
        var Lcircles = self.lines['left'].children();
        //var r = 1*(h-cy);
        for (var i in angs) {
            Rcircles[self.Ndots-i-1].attr({'cx':cx + 0.5*r*Math.cos(angs[i]),
                                         'cy':cy + 0.5*r*Math.sin(angs[i])});
            Lcircles[self.Ndots-i-1].attr({'cx':cx + r*Math.cos(angs[i]),
                                         'cy':cy + r*Math.sin(angs[i])});
        }
        self.spinRight = function () {
            self.lines['left'].attr({'transform':'r0,'+cx+','+cy});
            self.lines['right'].attr({'transform':'r0,'+cx+','+cy});
            self.lines['left'].animate({'transform':'r360,'+cx+','+cy}, 1500, mina.linear);
            self.lines['right'].animate({'transform':'r360,'+cx+','+cy}, 1500, mina.linear);
        };
        self.spinRight();
        self.rightSpinTimer = setInterval(self.spinRight, 1500);
    }).on('mouseleave', function (event) {
        clearTimeout(self.rightSpinTimer);
        self.lines['left'].stop();
        self.lines['right'].stop();
        self.lines['left'].attr({'transform':'r0', 'display':'none'});
        self.lines['right'].attr({'transform':'r0', 'display':'none'});
        self.lines['center'].attr({'display':'none'});
    });
    
    self.start = function () {           
        // everything i can think of to not get stuck driving...
        $(document).on("mouseleave.rfh mouseout.rfh", self.setUnsafe);
        $('.turn-signal').on('mouseleave.rfh mouseout.rfh mouseup.rfh blur.rfh', self.setUnsafe)
        $('.turn-signal').on('mousedown.rfh', self.driveGo);
        $(self.driveSVG.node, '.turn-signal').on('mouseleave.rfh mouseout.rfh mouseup.rfh blur.rfh', self.setUnsafe)
        $(self.driveSVG.node, '.turn-signal').on('mousedown.rfh', self.driveGo);
        $('.drive-ctrl').show();
        self.moveToStop(self.getNearestStop());
        $(self.driveSVG.node).on('resize.rfh', self.updateLineOffsets)
    }

    self.stop = function () {
        $(document).off("mouseleave.rfh mouseout.rfh");
        $('#'+self.div).removeClass('drive-safe');
        $(self.driveSVG.node, '.turn-signal').off('mouseleave.rfh mouseout.rfh mousedown.rfh mouseup.rfh hover')
        $('.drive-ctrl').hide();
    }

    self.driveGo = function (event) {
        clearTimeout(self.timer)
        if (event.which === 1) { //Only react to left mouse button
            self.setSafe();
            self.sendCmd(self.cmd);
        } else {
            self.setUnsafe();
        }
    }

    self.setSafe = function () {
        $('#'+self.div).addClass('drive-safe');
    }

    self.setUnsafe = function (event) {
        //alert("Unsafe: "+event.type);
        clearTimeout(self.timer);
        $('#'+self.div).removeClass('drive-safe');
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
//        console.log("R: "+r+", Theta: "+theta);
        return [r, theta, gnd_pt[0], gnd_pt[1]];
    }
    self.sendCmd = function (cmd) {
        if (!$('#'+self.div).hasClass('drive-safe')) { return };
        self.base.pubCmd(cmd.x, cmd.y, cmd.theta);
        self.timer = setTimeout(function(){self.sendCmd(self.cmd)}, 50);
    }
}




        //////////// DEBUG IMAGE OVERLAYS //////////////////////////
//        var imgC = self.camera.projectPoint(C[0], C[1], 0, 'base_link');
//        self.driveSVG.paper.circle(w*imgC[0], h*imgC[1], 5);
//        var imgB = self.camera.projectPoint(B[0], B[1], 0, 'base_link');
//        self.driveSVG.paper.circle(w*imgB[0], h*imgB[1], 5).attr({'fill':'blue'});
//        imgB = self.camera.projectPoint(BL[0], BL[1], 0, 'base_link');
//        self.driveSVG.paper.circle(w*imgB[0], h*imgB[1], 3).attr({'fill':'blue'});
//        imgB = self.camera.projectPoint(BR[0], BR[1], 0, 'base_link');
///        self.driveSVG.paper.circle(w*imgB[0], h*imgB[1], 3).attr({'fill':'blue'});
//        var imgT = self.camera.projectPoint(T[0], T[1], 0, 'base_link');
//        self.driveSVG.paper.circle(w*imgT[0], h*imgT[1], 5).attr({'fill':'red'});
//        /////////// END DEBUG ////////////////////////////////////////
//
/*    self.pseudoInv = function (A) {
        var aT = numeric.transpose(A);
        return numeric.dot(numeric.inv(numeric.dot(aT, A)), aT);
    };

    self.fitEllipse = function (pts) {
        // Fit Ellipse to data points. Following Numerically Stable Direct Lease Squares Fitting of Ellipses (Halir, Flusser, 1998)
        // Build Design Matrix D
        var D1 = [];
        var D2 = [];
        for (var i=0; i<pts.length; i += 1) {
            var x = pts[i][0];
            var y = pts[i][1];
            D1.push([x*x, x*y, y*y]);
            D2.push([x, y, 1]);
        }
        // Build Scatter Matrices S1-S3
        var D1T = numeric.transpose(D1);
        var D2T = numeric.transpose(D2);
        var S1 = numeric.dot(D1T, D1);
        var S2 = numeric.dot(D1T, D2);
        var S3 = numeric.dot(D2T, D2);
        
        var T = numeric.mul(-1, numeric.dot(numeric.inv(S3), numeric.transpose(S2)));
        var M = numeric.add(S1, numeric.dot(S2, T))
        var M = [numeric.mul(0.5, M[2]), numeric.mul(-1, M[1]), numeric.mul(0.5, M[0])];
        var eigs = numeric.eig(M);
        var eigVals = eigs.lambda.x;
        eigVals.push.apply(eigVals, eigs.lambda.y);
        var eigVecs = eigs.E.x;
        eigVecs.push.apply(eigVecs, eigs.E.y);
        var cond = numeric.mul(4, numeric.mul(eigVecs[0], eigVecs[2]));
        cond = numeric.sub(cond, numeric.pow(eigVecs[1], 2));
        var ind = numeric.gt(cond, 0).indexOf(true);
        var a1 = [eigVecs[0][ind], eigVecs[1][ind], eigVecs[2][ind]];
        var ellMatParams = a1;
        ellMatParams.push.apply(ellMatParams, numeric.dot(T, a1));

        var A = ellMatParams[0];
        var B = ellMatParams[1];
        var C = ellMatParams[2];
        var D = ellMatParams[3];
        var E = ellMatParams[4];
        var F = ellMatParams[5];
//        var Aq = [[A,   B/2, D/2],
//                  [B/2, C,   E/2],
//                  [D/2, E/2, F]];
//        var A33 = [[A,  B/2],
//                   [B/2, C]];
//        var I = A + C;
//        var detAq = numeric.det(Aq);
//        var detA33 = numeric.det(A33);
//        if (Math.abs(detAq) < numeric.epsilon) {
//            console.log("Degenerate Conic");
//            return;
//        } else {
//            if (detA33 > 0) {
//                if (detAq/I > 0) {
//                    console.log("Fit Imaginary Ellipse");
//                } else {
//                    console.log("Fit Real Ellipse");
//                }
//            } else if (detA33 < 0) {
//                console.log("Fit: Hyperbola")
//            } else if (Math.abs(detA33) < numeric.epsilon) {
//                console.log("Fit: Parabola");
//            }
//        }
        var Cx = (B*E - 2*C*D)/(4*A*C - B*B);
        var Cy = (D*B - 2*A*E)/(4*A*C - B*B);
//        self.driveSVG.paper.circle(Cx, Cy, 10);
        // Switch to Wolfram math page notation for simplicity...
        var B = B/2;
        var D = D/2;
        var E = E/2;
        var axis_B_len = Math.sqrt(2*(A*E*E+C*D*D+F*B*B-2*B*D*E-A*C*F)/((B*B-A*C)*(Math.sqrt((A-C)*(A-C)+4*B*B) - (A+C))));
        var axis_A_len = Math.sqrt(2*(A*E*E+C*D*D+F*B*B-2*B*D*E-A*C*F)/((B*B-A*C)*(-Math.sqrt((A-C)*(A-C)+4*B*B) - (A+C))));
        if (B < numeric.epsilon) {
            var rot = (A < C) ? 0 : Math.PI/2;
        } else {
            var rot = 0.5*Math.atan(2*B/(A-C));
            rot += (A > C) ? 0 : Math.PI/2;
        }
        self.ell = self.driveSVG.paper.ellipse(Cx, Cy, axis_A_len, axis_B_len).attr({'fill':'none',
                                                                                     'stroke':'purple'});
        var rotMat = new Snap.Matrix().rotate(rot*(180/Math.PI), Cx, Cy);
        self.ell.transform(rotMat.toTransformString());
        
    }
*/

 //   self.getLineParams = function (event) {
 //       var target = RFH.positionInElement(event); 
 //       var w = $(self.driveSVG.node).width();
 //       var h = $(self.driveSVG.node).height();
 //       var deadAhead = self.camera.projectPoint(1,0,0,'base_link');
 //       deadAhead[0] *= w;
 //       deadAhead[1] *= h;
 //       var originC = self.camera.projectPoint(0,0,0,'base_link');
 //       originC[0] *= w;
 //       originC[1] *= h;
 //       var fwdVec = [deadAhead[0] - originC[0], deadAhead[1] - originC[1]];
 //       var baselineVec = [-fwdVec[1], fwdVec[0]];
 //       var originL = self.camera.projectPoint(0,0.332,0,'base_link');
 //       originL[0] *= w;
 //       originL[1] *= h;
 //       var originR = self.camera.projectPoint(0,-0.332,0,'base_link');
 //       originR[0] *= w;
 //       originR[1] *= h;
 //       var tx = -(target[0]-originC[0]);
 //       var ty = -(target[1]-originC[1]);
 //       var ang = Math.atan2(tx, ty);
 //       var mag = Math.sqrt(tx*tx+ty*ty);
 //       var pathRad = 0.5*Math.abs(mag/Math.sin(ang));
 //       var cx = originC[0] + ((ang >= 0 ) ? -pathRad : pathRad) ;
 //       return {'cx':cx, 'originC':originC, 'originL':originL, 'originR':originR, 'rad':pathRad};
 //   };

//    self.getLineParams = function (event) {
//        var target = RFH.positionInElement(event); 
//        var originX = $('#drive-lines').width()/2 + self.lineCenterOffset;
//        var originY = $('#drive-lines').height();
//        var tx = -(target[0]-originX);
//        var ty = -(target[1]-originY);
//        var ang = Math.atan2(tx, ty);
//        var mag = Math.sqrt(tx*tx+ty*ty);
//        var pathRad = 0.5*Math.abs(mag/Math.sin(ang));
//        var cx = originX + ((ang >= 0 ) ? -pathRad : pathRad) ;
//        return {'cx':cx, 'originX':originX, 'originY':originY, 'rad':pathRad};
//    };

//    self.createLinesCB = function (event) {
//        var lp = self.getLineParams(event); 
//        if (lp.rad === Infinity) {
//            lp.rad = 10000
//            lp.cx = lp.rad + lp.originC[0];
//            }
//        //self.lines['center'] = self.driveSVG.path('M0,0').attr({
//        self.lines['center'] = self.driveSVG.circle(lp.cx, lp.originC[1], lp.rad).attr({
//                                    "id":"dl-line-center",
//                                    "stroke-width": 10,
//                                    "stroke-opacity":0.6,
//                                    "fill":"none"});
//        self.lines['left'] = self.lines['center'].clone().attr({
//                                    'id':'dl-line-left',
//                                    'rad':(lp.rad - lp.originL[0]-lp.originC[0]),
//                                    'cx':lp.cx});
//        self.lines['right'] = self.lines['center'].clone().attr({
//                                    'id':'dl-line-right',
//                                    'rad':(lp.rad + lp.originR[0]-lp.originC[0]),
//                                    'cx':lp.cx});
//    };
//    self.driveSVG.node.onmouseenter = self.createLinesCB;

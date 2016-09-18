var RFH = (function (module) {
    module.Drive = function (options) {
        "use strict";
        var self = this;
        options = options || {};
        self.name = options.name || 'drivingAction';
        self.buttonID = options.buttonID || 'drive-mode-button';
        self.forwardOnly = options.forwardOnly || false;
        self.showButton = true;
        self.toolTipText = "Drive the robot";
        self.currentStop = 'forward';
        var ros = options.ros;
        var tfClient = options.tfClient;
        var $viewer = $('#viewer-canvas');
        var head = options.head;
        var l_arm = options.left_arm;
        var r_arm = options.right_arm;
        var camera = options.camera;
        var base = options.base;
        var baseOffset = options.baseOffset || [0.0, 0.35]; //Half-width of PR2 Base [x - front/back, y-left/right]
        var timer = null;
        var spinTimer = null;
        var lines = {'left':null, 'center':null, 'right':null};
        var nDots = 25;
        var driveSVG = new Snap('#drive-lines');
        self.$div = $(driveSVG.node);
        var headTrackingTimer = null;

        self.baseContactDisplay = new module.BumperDisplay({tfClient: tfClient,
                                                            head: head,
                                                            camera: camera,
                                                            skins: [RFH.skins.base]
        });

        self.goalDisplay = new RFH.DriveGoalDisplay({
            ros: ros,
            tfClient: tfClient,
            viewer: $viewer
        });

        var clamp = function (x,a,b) {
            return ( x < a ) ? a : ( ( x > b ) ? b : x );
        };

        self.showGoal = self.goalDisplay.show;
        self.hideGoal = self.goalDisplay.hide;

        var tuckActionClient = new ROSLIB.ActionClient({
            ros: ros, 
            serverName: 'tuck_arms',
            actionName: 'pr2_common_action_msgs/TuckArmsAction'
        });

        ros.getMsgDetails('pr2_common_action_msgs/TuckArmsGoal');
        var tuckArms = function (event) {
            var goal_msg = ros.composeMsg('pr2_common_action_msgs/TuckArmsGoal');
            goal_msg.tuck_left = true;
            goal_msg.tuck_right = true;
            var goal = new ROSLIB.Goal({
                actionClient: tuckActionClient,
                goalMessage: goal_msg
            });
            var resultCB = function (result) {
                console.log("Tuck Arms Completed, re-enabling hapticMPC");
                l_arm.enableMPC();
                r_arm.enableMPC();
            };
            goal.on('result', resultCB);
            var sendGoalOnceDisabled = function (resp) {
                goal.send();
                //            setTimeout(resultCB, 18000);
            };
            l_arm.disableMPC();
            r_arm.disableMPC(sendGoalOnceDisabled);

        };
        $('#controls > div.tuck-driving.drive-ctrl').button().on('click.rfh', tuckArms);

        var initPathMarkers = function (nDots, d) {
            var opacity = numeric.linspace(1, 0.05, nDots);
            lines.left = driveSVG.g();
            lines.center = driveSVG.g();
            lines.right = driveSVG.g();
            for (var i=0; i<nDots; i += 1) {
                var c = driveSVG.paper.circle(0, 0, d);
                c.attr({'fill-opacity':opacity[i]});
                lines.left.add(c);
                lines.right.add(c.clone());
                lines.center.add(c.clone());
            }
            lines.left.attr({'fill':'rgb(0,188,212)'});
            lines.right.attr({'fill':'rgb(0,188,212)'});
            lines.center.attr({'fill':'rgb(33,150,243)'});
        }(nDots, 4.5); // Define and call initialization

        var showLinesCB = function (event) {
            lines.left.attr({'display':'block'});
            lines.right.attr({'display':'block'});
            lines.center.attr({'display':'block'});
        };
        self.$div.on('mouseenter.rfh', showLinesCB).css('zIndex', 5);

        var removeLinesCB = function (event) {
            if ($(event.relatedTarget).hasClass('turn-signal')){ return; }
            lines.left.attr({'display':'none'});
            lines.right.attr({'display':'none'});
            lines.center.attr({'display':'none'});
        };
        self.$div.on('mouseleave.rfh', removeLinesCB);

        var getWorldPath = function (rtxy) {
            var T = rtxy.slice(2); //Target (clicked) Point on floor in real world
            var B = [0, 0]; //origin of motion (base movement moves center of robot)
            var A = [0, 0]; //point straight ahead.
            var BL = [0, 0]; //Left base line point
            var BR = [0, 0]; //Right base line point
            var C = [0, 0]; // Center of circle defined by base center and clicked target point
            var side = '';
            var dirAngleOffset;
            switch (self.currentStop) {
                case 'forward':
                    A = [B[0] + 1, B[1]];
                    BL = [B[0] + baseOffset[0], B[1] + baseOffset[1]]; //Left and right guide-line reference points
                    BR = [B[0] + baseOffset[0], B[1] - baseOffset[1]]; 
                    C[0] = B[0]; // Center is on line with base center
                    //Center is equidistant from base center to target -- Must for corner of a square with base and target.
                    C[1] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[0]*(B[0] - T[0]) ) / (2 * (B[1] - T[1]));
                    side = (T[1] > B[1]) ? "left" : "right";
                    dirAngleOffset = 0;
                    break;
                case 'back-left':
                case 'back-right':
                    A = [B[0] - 1, B[1]];
                    BL = [B[0] - baseOffset[0], B[1] - baseOffset[1]];
                    BR = [B[0] - baseOffset[0], B[1] + baseOffset[1]];
                    C[0] = B[0]; // Center is on line with base center
                    C[1] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[0]*(B[0] - T[0]) ) / (2 * (B[1] - T[1]));
                    side = (T[1] < B[1]) ? "left" : "right";
                    dirAngleOffset = Math.PI;
                    break;
                case 'left':
                    A = [B[0], B[1] + 1];
                    BL = [B[0] - baseOffset[1], B[1] + baseOffset[0]];
                    BR = [B[0] + baseOffset[1], B[1] + baseOffset[0]];
                    C[1] = B[1];
                    C[0] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[1]*(B[1] - T[1]) ) / (2 * (B[0] - T[0]));
                    side = (T[0] < B[0]) ? "left" : "right";
                    dirAngleOffset = Math.PI/2;
                    break;
                case 'right':
                    A = [B[0], B[1] - 1];
                    BL = [B[0] + baseOffset[1], B[1] - baseOffset[0]];
                    BR = [B[0] - baseOffset[1], B[1] - baseOffset[0]];
                    C[1] = B[1];
                    C[0] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[1]*(B[1] - T[1]) ) / (2 * (B[0] - T[0])); 
                    side = (T[0] > B[0]) ? "left" : "right";
                    dirAngleOffset = -Math.PI/2;
                    break;
            }
            var R = Math.sqrt( Math.pow(C[0]-B[0], 2) + Math.pow(C[1]-B[1], 2) ); // Radius of the defined circle
            var RL = Math.sqrt( Math.pow(C[0]-BL[0], 2) + Math.pow(C[1]-BL[1], 2) ); // Radius of circle to left edge point
            var RR = Math.sqrt( Math.pow(C[0]-BR[0], 2) + Math.pow(C[1]-BR[1], 2) ); // Radius of circle to right edge point
            var CB = numeric.sub(C,B); 
            var CT = numeric.sub(C,T);
            var theta = Math.acos(numeric.dot(CB, CT)/(R*R));  // Arc of circle between base center and target points
            return {'R':R, 'RL':RL, 'RR':RR, 'side':side, 'dirAngleOffset': dirAngleOffset, 'C':C, 'B':B, 'T':T, 'theta':theta};
        };

        var drawPath = function (rtxy) {
            // Find points around the circle from start to target location.
            var path = getWorldPath(rtxy);
            self.cmd = calcCmd(path);
            path.theta *= 1.4; // Extend path past point targeted by mouse
            var nDots = 25;
            var minAng = (path.side === 'right') ? Math.PI/2 + path.dirAngleOffset : -Math.PI/2 + path.dirAngleOffset;
            var maxAng = (path.side === 'right') ? minAng - path.theta : minAng + path.theta;
            var angs = numeric.linspace(minAng, maxAng, nDots);
            var opacity = numeric.linspace(1, 0.05, nDots);
            var w = self.$div.width();
            var h = self.$div.height();
            var pts = [];
            var imgpts;
            var i;

            for (i in angs) {
                pts.push([path.C[0] + path.R*Math.cos(angs[i]), path.C[1] + path.R*Math.sin(angs[i]), 0]);
            }
            imgpts = camera.projectPoints(pts, 'base_link');
            for (i=0; i<imgpts.length; i += 1) {
                imgpts[i][0] *= w;
                imgpts[i][1] *= h;
                lines.center[i].attr({'cx':imgpts[i][0], 'cy':imgpts[i][1]});
            }

            if (path.side === 'right' || path.RL < path.R) {
                pts = [];
                for (i in angs) {
                    pts.push([path.C[0] + path.RL*Math.cos(angs[i]), path.C[1] + path.RL*Math.sin(angs[i]), 0]);
                }
                imgpts = camera.projectPoints(pts, 'base_link');
                //Draw points
                for (i=0; i<imgpts.length; i += 1) {
                    imgpts[i][0] *= w;
                    imgpts[i][1] *= h;
                    lines.left[i].attr({'cx':imgpts[i][0], 'cy':imgpts[i][1]});
                }
            }

            if (path.side == 'left' ||  path.RR < path.R) {
                pts = [];
                for (i in angs) {
                    pts.push([path.C[0] + path.RR*Math.cos(angs[i]), path.C[1] + path.RR*Math.sin(angs[i]), 0]);
                }
                imgpts = camera.projectPoints(pts, 'base_link');
                //Draw points
                for (i=0; i<imgpts.length; i += 1) {
                    imgpts[i][0] *= w;
                    imgpts[i][1] *= h;
                    lines.right[i].attr({'cx':imgpts[i][0], 'cy':imgpts[i][1]});
                }
            }
        };

        var calcCmd = function (path) {
            var pathLength = path.R * path.theta;
            var targetDist = Math.sqrt( Math.pow(path.T[0]-path.B[0], 2) + Math.pow(path.T[1]-path.B[1], 2) );
            var linVel = clamp(0.2*targetDist, 0.0, 0.2);
            var cmd_theta = Math.tan(linVel/path.R);
            cmd_theta *= (path.side === 'left') ? 1 : -1;
            var cmd_x = 0;
            var cmd_y = 0;
            switch (self.currentStop) {
                case "forward":
                    cmd_x = linVel;
                    break;
                case "back-left":
                case "back-right":
                    cmd_x = -linVel;
                    break;
                case "left":
                    cmd_y = linVel;
                    break;
                case "right":
                    cmd_y = -linVel;
                    break;
            }
            return {'x':cmd_x, 'y':cmd_y, 'theta':cmd_theta};
        };

        var drawSlidePath = function (rtxy) {
            var B, L, R;
            switch (self.currentStop) {
                case "forward":
                    B = [0.2, 0, 0];
                    L = [0.33*B[0]+baseOffset[0], B[1] + baseOffset[1], 0];
                    R = [0.33*B[0]+baseOffset[0], B[1] - baseOffset[1], 0];
                    break;
                case "back-left":
                case "back-right":
                    B = [-0.2, 0, 0];
                    L = [-0.33*B[0]+baseOffset[0], B[1] + baseOffset[1], 0];
                    R = [-0.33*B[0]+baseOffset[0], B[1] - baseOffset[1], 0];
                    break;
                case "left":
                    B = [0, 0.2, 0];
                    L = [B[0]-baseOffset[1], 0.5*B[1] + baseOffset[0], 0];
                    R = [B[0]+baseOffset[1], 0.5*B[1] + baseOffset[0], 0];
                    break;
                case "right":
                    B = [0, -0.2, 0];
                    L = [B[0]+baseOffset[1], 0.5*B[1] - baseOffset[0], 0];
                    R = [B[0]-baseOffset[1], 0.5*B[1] - baseOffset[0], 0];
                    break;
            }
            var T = rtxy.slice(2); //Target (clicked) Point on floor in real world
            var pts = [B, [T[0], T[1], 0], L, R];
            var imgpts = camera.projectPoints(pts, 'base_link');
            var w = self.$div.width();
            var h = self.$div.height();
            var i;
            for (i=0; i<imgpts.length; i += 1) {
                imgpts[i][0] *= w;
                imgpts[i][1] *= h;
            }
            var slide = [imgpts[1][0] - imgpts[0][0], imgpts[1][1] - imgpts[0][1]];
            var slideX = numeric.linspace(0, slide[0], nDots);
            var slideY = numeric.linspace(0, slide[1], nDots);
            for (i=0; i<nDots; i += 1) {
                lines.left[i].attr({'cx':imgpts[2][0] + slideX[i], 'cy':imgpts[2][1] + slideY[i]});
                lines.center[i].attr({'cx':imgpts[0][0] + slideX[i], 'cy':imgpts[0][1] + slideY[i]});
                lines.right[i].attr({'cx':imgpts[3][0] + slideX[i], 'cy':imgpts[3][1] + slideY[i]});
            }
            var delX = clamp((T[0] - B[0])/2, -0.45, 0.45);
            var delY = clamp((T[1] - B[1])/2, -0.45, 0.45);
            self.cmd = {'x':delX, 'y':delY, 'theta':0};
        };

        self.updateVis = function (event) {
            var rtxy = self.getRTheta(event); //Get real-world point in base frame
            if (rtxy[0] < 0.45) {
                drawSlidePath(rtxy);
            } else {
                drawPath(rtxy);
            }
        };
        self.$div.on("mousemove.rfh", self.updateVis);

        var headStops = ['back-left', 'left', 'forward', 'right', 'back-right'];
        var headStopAngles = {'back-right': [-2.85, 1.35],
            'right': [-Math.PI/2, 1.35],
            'forward': [0.0, 1.35],
            'left': [Math.PI/2, 1.35],
            'back-left': [2.85, 1.35]};

        var toLeft = function (e) {
            var newStop = headStops[headStops.indexOf(self.currentStop) - 1];
            if (newStop) {
                trackHeadPosition(newStop);
            }
        };

        var toRight = function (e) {
            var newStop = headStops[headStops.indexOf(self.currentStop) + 1];
            if (newStop) {
                trackHeadPosition(newStop);
            }
        };
        //    $('.drive-look.left').on('click.rfh', toLeft).prop('title', 'Turn head left one step.');
        //    $('.drive-look.right').on('click.rfh', toRight).prop('title', 'Turn head right one step.');

        var getNearestStop = function () {
            if (self.forwardOnly) { return headStops[2]; }
            var currentPan = head.getState()[0];
            var nearestStop = 'forward'; //Good default assumption;
            var del = 2*Math.PI; //Initialize too high;
            for (var i=0; i < headStops.length; i++ ) {
                var stop = headStops[i];
                var dist = Math.abs(headStopAngles[stop][0] - currentPan);
                if (dist <= del) {
                    del = dist;
                    nearestStop = stop;
                }
            }
            return nearestStop;
        };

        var moveToStop = function (stopName) {
            var angs = headStopAngles[stopName];
            self.currentStop = stopName;
            $('#drive-dir-icon path').css({'fill':'rgb(30,220,250)'});
            $('#drive-dir-icon path.'+stopName).css({'fill':'#ffffff'});
            head.setPosition(angs[0], angs[1]);
        };

        var trackHeadPosition = function (stopName) {
            var angs = headStopAngles[stopName];
            head.trackAngles(angs[0], angs[1]);
        };
        var stopTracking = head.stopTracking;

        if (!self.forwardOnly) {
            var driveDirIcon = new Snap("#drive-dir-icon");
            Snap.load('./css/icons/drive-direction-icon.svg', function (icon_svg) {
                driveDirIcon.append(icon_svg.select('g'));
                var wedgeClickCB = function (e) {
                    trackHeadPosition(e.target.classList[0]);
                };
                var wedges = driveDirIcon.selectAll('path');
                for (var i=0; i<wedges.length; i+=1) {
                    wedges[i].click(wedgeClickCB);
                }
                console.log("Drive Direction Icon Loaded"); 
            });
        } else {
            $('#drive-dir-icon').remove();
        }

        var linesSpin = function (cx, cy, dir, hz) {
            hz = hz || 1500;
            lines.left.attr({'transform':'r0,'+cx+','+cy});
            lines.right.attr({'transform':'r0,'+cx+','+cy});
            if (dir[0] == 'l') { // left
                lines.left.animate({'transform':'r-360,'+cx+','+cy}, hz, mina.linear);
                lines.right.animate({'transform':'r-360,'+cx+','+cy}, hz, mina.linear);
            } else {
                lines.left.animate({'transform':'r360,'+cx+','+cy}, hz, mina.linear);
                lines.right.animate({'transform':'r360,'+cx+','+cy}, hz, mina.linear);
            }
        };

        $('.turn-signal.left').on('mouseenter', function (event) {
            self.cmd = {'x':0, 'y':0, 'theta':0.1*Math.PI};
            lines.left.attr({'display':'block'});
            lines.right.attr({'display': 'block'});
            lines.center.attr({'display':'block'});
            var w = self.$div.width();
            var h = self.$div.height();
            var pts = camera.projectPoints([[0,0,0],[0,0.2,0]],'base_link');
            var cx = pts[0][0]*w;
            var cy = pts[0][1]*h;
            var r = numeric.norm2(numeric.sub(pts[1], pts[0]))*w;
            lines.center.children().forEach(function(c){c.attr({'cx':cx, 'cy':cy});});
            var angs = numeric.linspace(-Math.PI, 0.85*Math.PI, nDots);
            var Rcircles = lines.right.children();
            var Lcircles = lines.left.children();
            for (var i in angs) {
                Rcircles[i].attr({'cx':cx + 0.5*r*Math.cos(angs[i]),
                    'cy':cy + 0.5*r*Math.sin(angs[i])});
                Lcircles[i].attr({'cx':cx + r*Math.cos(angs[i]),
                    'cy':cy + r*Math.sin(angs[i])});
            }
            linesSpin(cx, cy, 'left', 1500);
            spinTimer = setInterval(function(){linesSpin(cx, cy, 'left', 1500);}, 1500);
        }).on('mouseleave', function (event) {
            clearTimeout(spinTimer);
            lines.left.stop();
            lines.right.stop();
            lines.left.attr({'transform':'r0', 'display':'none'});
            lines.right.attr({'transform':'r0', 'display':'none'});
            lines.center.attr({'display':'none'});
        });
        $('.turn-signal.left').prop('title', 'Rotate left\n(in place)');

        $('.turn-signal.right').on('mouseenter', function (event) {
            self.cmd = {'x':0, 'y':0, 'theta':-0.1*Math.PI};
            lines.left.attr({'display':'block'});
            lines.right.attr({'display': 'block'});
            lines.center.attr({'display':'block'});
            var w = self.$div.width();
            var h = self.$div.height();
            var pts = camera.projectPoints([[0,0,0],[0,0.2,0]],'base_link');
            var cx = pts[0][0]*w;
            var cy = pts[0][1]*h;
            var r = numeric.norm2(numeric.sub(pts[1], pts[0]))*w;
            lines.center.children().forEach(function(c){c.attr({'cx':cx, 'cy':cy});});
            var angs = numeric.linspace(-Math.PI, 0.85*Math.PI, nDots);
            var Rcircles = lines.right.children();
            var Lcircles = lines.left.children();
            for (var i in angs) {
                Rcircles[nDots-i-1].attr({'cx':cx + 0.5*r*Math.cos(angs[i]),
                    'cy':cy + 0.5*r*Math.sin(angs[i])});
                Lcircles[nDots-i-1].attr({'cx':cx + r*Math.cos(angs[i]),
                    'cy':cy + r*Math.sin(angs[i])});
            }
            linesSpin(cx, cy, 'right', 1500);
            spinTimer = setInterval(function(){linesSpin(cx, cy, 'right', 1500);}, 1500);
        }).on('mouseleave', function (event) {
            clearTimeout(spinTimer);
            lines.left.stop();
            lines.right.stop();
            lines.left.attr({'transform':'r0', 'display':'none'});
            lines.right.attr({'transform':'r0', 'display':'none'});
            lines.center.attr({'display':'none'});
        });
        $('.turn-signal.right').prop('title', 'Rotate right\n(in place)');


        self.driveGo = function (event) {
//            console.log("Clearing timer for new cmd");
            clearTimeout(timer);
            if (event.which === 1) { //Only react to left mouse button
                self.setSafe();
                self.sendCmd(self.cmd);
            } else {
                self.setUnsafe(event);
            }
            event.stopPropagation();
        };

        self.setSafe = function () {
            self.$div.addClass('drive-safe');
        };

        self.setUnsafe = function (event) {
            //alert("Unsafe: "+event.type);
//            console.log(event.type.toString() + ": Clearing timer to stop driving");
            clearTimeout(timer);
            self.$div.removeClass('drive-safe');
        };

        self.getRTheta = function (e) {
            var $target = $(e.target);
            var target_pos = $target.position();
            var target_width = $target.width();
            var target_height = $target.height();
            var pt = [e.pageX - target_pos.left, e.pageY - target_pos.top];
            var px = (pt[0]/target_width) * camera.width;
            var py = (pt[1]/target_height) * camera.height;
            if (camera.frame_id === '') {
                alert("Camera position not up to date.  Cannot drive safely.");
                camera.updateCameraInfo();
            }
            var xyz = camera.projectPixel(px, py, 1.0);
            var pose = new ROSLIB.Pose({position:{x: xyz[0],
                y: xyz[1], 
                z: xyz[2]}});
            pose.applyTransform(camera.transform);
            if (pose.position.z >= camera.transform.translation.z) {
                RFH.log('Please click on the ground near the robot to drive.');
                throw new Error("Clicked point not on the ground");
            }
            var z0 = camera.transform.translation.z;
            var z1 = pose.position.z;
            var dist = (z0+0.05)/(z0-z1); // -0.05 = z0 - ((z0-z1)/1)*x -> lenght of line to intersection
            var gnd_pt = [0,0,0];
            gnd_pt[0] = camera.transform.translation.x + (pose.position.x - camera.transform.translation.x) * dist;
            gnd_pt[1] = camera.transform.translation.y + (pose.position.y - camera.transform.translation.y) * dist; 
            var r = Math.sqrt(gnd_pt[0]*gnd_pt[0] + gnd_pt[1]*gnd_pt[1]);
            var theta = Math.atan2(gnd_pt[1], gnd_pt[0]);
            //        console.log("R: "+r+", Theta: "+theta);
            return [r, theta, gnd_pt[0], gnd_pt[1]];
        };

        self.sendCmd = function (cmd) {
//            console.log("Prepared to send ", cmd);
            if (!self.$div.hasClass('drive-safe')) {
                console.log("Not safe to drive"); 
                return ;}
            base.pubCmd(cmd.x, cmd.y, cmd.theta);
//            console.log("Sent ", cmd);
            timer = setTimeout(function(){self.sendCmd(self.cmd);}, 50);
            console.log("Set Timer: ", timer);
        };

        $(document).on("mouseleave.rfh mouseout.rfh", self.setUnsafe);
        $('.turn-signal').on('mouseleave.rfh mouseout.rfh mouseup.rfh blur.rfh', self.setUnsafe);
        $('.turn-signal').on('mouseenter.rfh', function(){self.$div.blur();}); // Blurs driving on enter.  Otherwise, blur occurs on 1st click, and blur cb sets unsafe, stopping turning.

        self.start = function () {           
            // everything i can think of to not get stuck driving...
            $('.turn-signal').on('mousedown.rfh', self.driveGo);
            self.$div.on('mouseleave.rfh mouseout.rfh mouseup.rfh blur.rfh', self.setUnsafe);
            self.$div.on('mousedown.rfh', self.driveGo);
            $('.drive-ctrl').show();
            self.showGoal();
            $viewer.show();
            self.baseContactDisplay.show();
            trackHeadPosition(getNearestStop());
            self.$div.on('resize.rfh', self.updateLineOffsets);
            $('#controls h3').text("Head Controls");
        };

        self.stop = function () {
            //   $(document).off("mouseleave.rfh mouseout.rfh");
            self.$div.removeClass('drive-safe');
            //   self.$div'.turn-signal').off('mouseleave.rfh mouseout.rfh mousedown.rfh mouseup.rfh hover');
            self.hideGoal();
            $('.drive-ctrl').hide();
            $viewer.hide();
            self.baseContactDisplay.hide();
            $('#controls h3').text("Controls");
            stopTracking();
        };
    };
    return module;
})(RFH || {});

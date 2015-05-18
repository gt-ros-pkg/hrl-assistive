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
    self.baseOffset = [0.35, 0.1]; //Half-width of PR2 Base [l/r, f/b]
    self.lineWidthOffset = 0.18; //5% image width offset between lines
    self.clamp = function (x,a,b) {
        return ( x < a ) ? a : ( ( x > b ) ? b : x );
    }
    self.sign = function (x) { 
        return typeof x === 'number' ? x ? x < 0 ? -1 : 1 : x === x ? 0 : NaN : NaN;
    }
   
    self.driveSVG = Snap('#drive-lines');

    self.getLineParams = function (event) {
        var target = RFH.positionInElement(event); 
        var w = $(self.driveSVG.node).width();
        var h = $(self.driveSVG.node).height();
        var deadAhead = self.camera.projectPoint(1,0,0,'base_link');
        deadAhead[0] *= w;
        deadAhead[1] *= h;
        var originC = self.camera.projectPoint(0,0,0,'base_link');
        originC[0] *= w;
        originC[1] *= h;
        var fwdVec = [deadAhead[0] - originC[0], deadAhead[1] - originC[1]];
        var baselineVec = [-fwdVec[1], fwdVec[0]];
        var originL = self.camera.projectPoint(0,0.332,0,'base_link');
        originL[0] *= w;
        originL[1] *= h;
        var originR = self.camera.projectPoint(0,-0.332,0,'base_link');
        originR[0] *= w;
        originR[1] *= h;
        var tx = -(target[0]-originC[0]);
        var ty = -(target[1]-originC[1]);
        var ang = Math.atan2(tx, ty);
        var mag = Math.sqrt(tx*tx+ty*ty);
        var pathRad = 0.5*Math.abs(mag/Math.sin(ang));
        var cx = originC[0] + ((ang >= 0 ) ? -pathRad : pathRad) ;
        return {'cx':cx, 'originC':originC, 'originL':originL, 'originR':originR, 'rad':pathRad};
    };

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

    self.createLinesCB = function (event) {
        var lp = self.getLineParams(event); 
        if (lp.rad === Infinity) {
            lp.rad = 10000
            lp.cx = lp.rad + lp.originC[0];
            }
        //self.lines['center'] = self.driveSVG.path('M0,0').attr({
        self.lines['center'] = self.driveSVG.circle(lp.cx, lp.originC[1], lp.rad).attr({
                                    "id":"dl-line-center",
                                    "stroke-width": 10,
                                    "stroke-opacity":0.6,
                                    "fill":"none"});
        self.lines['left'] = self.lines['center'].clone().attr({
                                    'id':'dl-line-left',
                                    'rad':(lp.rad - lp.originL[0]-lp.originC[0]),
                                    'cx':lp.cx});
        self.lines['right'] = self.lines['center'].clone().attr({
                                    'id':'dl-line-right',
                                    'rad':(lp.rad + lp.originR[0]-lp.originC[0]),
                                    'cx':lp.cx});
    };
    self.driveSVG.node.onmouseenter = self.createLinesCB;

    self.removeLinesCB = function (event) {
        self.driveSVG.paper.clear();
//        self.lines['center'].remove();
//        self.lines['left'].remove();
//        self.lines['right'].remove();
        self.lines = {};
    };
    self.driveSVG.node.onmouseleave = self.removeLinesCB;

    self.updateLinesCB = function (event) {
        var lp = self.getLineParams(event);
        if (lp.rad === Infinity) { return }
        self.lines['center'].attr({'cx':lp.cx, 'cy':lp.originC[1], 'r': lp.rad});
        self.lines['left'].attr({'cx':lp.cx, 'cy':lp.originC[1], 'r': Math.max((lp.rad-lp.originL[0]-lp.originC[0]), 0)});
        self.lines['right'].attr({'cx':lp.cx, 'cy':lp.originC[1], 'r': Math.max((lp.rad+lp.originR[0]-lp.originC[0]), 0)});
    };
//    self.driveSVG.node.onmousemove = self.updateLinesCB;

 
    self.getWorldPath = function (event) {
        if (event.target.id !== 'drive-lines') { return };
        self.driveSVG.paper.clear();
        var rtxy = self.getRTheta(event); //Get real-world point in base frame
        var T = rtxy.slice(2); //Target (clicked) Point on floor in real world
        var B = [0, 0]; //origin of motion (base movement moves center of robot)
        var A = new Array(2); //point straight ahead.
        var BL = new Array(2); //Left base line point
        var BR = new Array(2); //Right base line point
        var C = new Array(2);
        var side = '';
        switch (self.currentStop) {
            case 'forward':
                A = [B[0] + 1, B[1]];
                BL = [B[0] + self.baseOffset[1], B[1] + self.baseOffset[0]] //Left and right guide-line reference points
                BR = [B[0] + self.baseOffset[1], B[1] - self.baseOffset[0]] 
                C[0] = B[0]; // Center is on line with base center
                //Center is equidistant from base center to target -- Must for corner of a square with base and target.
                C[1] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[0]*(B[0] - T[0]) ) / (2 * (B[1] - T[1]));
                side = (T[1] > B[1]) ? "left" : "right";
                break;
            case 'back-left':
            case 'back-right':
                A = [B[0] - 1, B[1]];
                BL = [B[0] - self.baseOffset[1], B[1] - self.baseOffset[0]]
                BR = [B[0] - self.baseOffset[1], B[1] + self.baseOffset[0]]
                C[0] = B[0]; // Center is on line with base center
                C[1] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[0]*(B[0] - T[0]) ) / (2 * (B[1] - T[1]));
                side = (T[1] < B[1]) ? "left" : "right";
                break;
            case 'left':
                A = [B[0], B[1] + 1];
                BL = [B[0] - self.baseOffset[0], B[1] + self.baseOffset[1]]
                BR = [B[0] + self.baseOffset[0], B[1] + self.baseOffset[1]]
                C[1] = B[1];
                C[0] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[1]*(B[1] - T[1]) ) / (2 * (B[0] - T[0]));
                side = (T[0] < B[0]) ? "left" : "right";
                break;
            case 'right':
                A = [B[0], B[1] - 1];
                BL = [B[0] + self.baseOffset[0], B[1] - self.baseOffset[1]]
                BR = [B[0] - self.baseOffset[0], B[1] - self.baseOffset[1]]
                C[1] = B[1];
                C[0] = ( B[0]*B[0] + B[1]*B[1] - T[0]*T[0] - T[1]*T[1] - 2 * C[1]*(B[1] - T[1]) ) / (2 * (B[0] - T[0])); 
                side = (T[0] > B[0]) ? "left" : "right";
                break;
        }
        var R = Math.sqrt( Math.pow(C[0]-B[0], 2) + Math.pow(C[1]-B[1], 2) )
        var RL = Math.sqrt( Math.pow(C[0]-BL[0], 2) + Math.pow(C[1]-BL[1], 2) )
        var RR = Math.sqrt( Math.pow(C[0]-BR[0], 2) + Math.pow(C[1]-BR[1], 2) )

        // Find points around the circle from start to target location.
        var w = $(self.driveSVG.node).width();
        var h = $(self.driveSVG.node).height();

        //////////// DEBUG IMAGE OVERLAYS //////////////////////////
        var imgC = self.camera.projectPoint(C[0], C[1], 0, 'base_link');
        self.driveSVG.paper.circle(w*imgC[0], h*imgC[1], 10);
        var imgB = self.camera.projectPoint(B[0], B[1], 0, 'base_link');
        self.driveSVG.paper.circle(w*imgB[0], h*imgB[1], 10).attr({'fill':'blue'});
        imgB = self.camera.projectPoint(BL[0], BL[1], 0, 'base_link');
        self.driveSVG.paper.circle(w*imgB[0], h*imgB[1], 10).attr({'fill':'blue'});
        imgB = self.camera.projectPoint(BR[0], BR[1], 0, 'base_link');
        self.driveSVG.paper.circle(w*imgB[0], h*imgB[1], 10).attr({'fill':'blue'});
//        var imgT = self.camera.projectPoint(T[0], T[1], 0, 'base_link');
//        self.driveSVG.paper.circle(w*imgT[0], h*imgT[1], 10).attr({'fill':'red'});
        /////////// END DEBUG ////////////////////////////////////////

        var pts = [];
        var angs = numeric.linspace(-Math.PI, 3*Math.PI/4, 6);
        for (var i in angs) {
            pts.push([C[0] + R*Math.cos(angs[i]), C[1] + R*Math.sin(angs[i]), 0]);
        }
        var imgptsC = self.camera.projectPoints(pts, 'base_link');
        for (var i=0; i<imgptsC.length; i += 1) {
            imgptsC[i][0] *= w;
            imgptsC[i][1] *= h;
            self.driveSVG.paper.circle(imgptsC[i][0], imgptsC[i][1], 4).attr({'fill':'green'});
        }
        var ell_C = self.fitEllipse(imgptsC);

        if (side === 'right' || RL < R) {
            pts = [];
            for (var i in angs) {
                pts.push([C[0] + RL*Math.cos(angs[i]), C[1] + RL*Math.sin(angs[i]), 0]);
            }
            var imgptsL = self.camera.projectPoints(pts, 'base_link');
            //Draw points
            for (var i=0; i<imgptsL.length; i += 1) {
                imgptsL[i][0] *= w;
                imgptsL[i][1] *= h;
                self.driveSVG.paper.circle(imgptsL[i][0], imgptsL[i][1], 4).attr({'fill':'red'});
            }
//            var ell_L = self.fitEllipse(imgptsL);
        }

        if (side == 'left' ||  RR < R) {
            pts = [];
            for (var i in angs) {
                pts.push([C[0] + RR*Math.cos(angs[i]), C[1] + RR*Math.sin(angs[i]), 0]);
            }
            var imgptsR = self.camera.projectPoints(pts, 'base_link');
            //Draw points
            for (var i=0; i<imgptsR.length; i += 1) {
                imgptsR[i][0] *= w;
                imgptsR[i][1] *= h;
                self.driveSVG.paper.circle(imgptsR[i][0], imgptsR[i][1], 4).attr({'fill':'blue'});
            }
//            var ell_R = self.fitEllipse(imgptsR);
        }
    };

    self.pseudoInv = function (A) {
        var aT = numeric.transpose(A);
        return numeric.dot(numeric.inv(numeric.dot(aT, A)), aT);
    };

    self.fitEllipse = function (pts) {
        // Fit Ellipse to data points. Following Numerically Stable Direct Lease Squares Fitting of Ellipses (Halir, Flusser, 1998)
        //Make test points;
        var Center = [0,0];
        var h = $(self.driveSVG.node).height();
        var w = $(self.driveSVG.node).width();
        var R = 150;
        Center[0] = w/2;
        Center[1] = h/2;
        pts = [];
        var angs = numeric.linspace(-Math.PI, (5/6)*Math.PI, 20);
        for (var i in angs) {
            pts.push([Center[0] + (0.49+(0.02*Math.random()))*R*Math.cos(angs[i]), Center[1] + (0.99+(0.02*Math.random()))*R*Math.sin(angs[i])]);
        }
        for (var i in pts) {
            self.driveSVG.paper.circle(pts[i][0], pts[i][1], 4).attr({'fill':'brown'});
        }
        // Get points as relative to centroid;
        //var centroid =[0,0];
        //for (var i in pts) {
        //    centroid[0] += pts[i][0];
        //    centroid[1] += pts[i][1];
        //}
        //centroid[0] = centroid[0]/pts.length;
        //centroid[1] = centroid[1]/pts.length;
        //for (var i in pts) {
        //    pts[i][0] -= centroid[0];
        //    pts[i][1] -= centroid[1];
        //}

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
        var a1 = eigVecs[numeric.gt(cond, 0).indexOf(true)];
        var ellMatParams = a1;
        ellMatParams.push.apply(ellMatParams, numeric.dot(T, a1));

        var A = ellMatParams[0];
        var B = ellMatParams[1];
        var C = ellMatParams[2];
        var D = ellMatParams[3];
        var E = ellMatParams[4];
        var F = ellMatParams[5];
        var Aq = [[A,   B/2, D/2],
                  [B/2, C,   E/2],
                  [D/2, E/2, F]];
        var A33 = [[A,  B/2],
                   [B/2, C]];
        var I = A + C;
        var detAq = numeric.det(Aq);
        var detA33 = numeric.det(A33);
        if (Math.abs(detAq) < numeric.epsilon) {
            console.log("Degenerate Conic");
            return;
        } else {
            if (detA33 > 0) {
                if (detAq/I > 0) {
                    console.log("Fit Imaginary Ellipse");
                } else {
                    console.log("Fit Real Ellipse");
                }
            } else if (detA33 < 0) {
                console.log("Fit: Hyperbola")
            } else if (Math.abs(detA33) < numeric.epsilon) {
                console.log("Fit: Parabola");
            }
        }
        var Cx = (B*E - 2*C*D)/(4*A*C - B*B);
        var Cy = (D*B - 2*A*E)/(4*A*C - B*B);
        self.driveSVG.paper.circle(Cx, Cy, 10);

        // Switch to Wolfram math page notation for simplicity...
        var a = A;
        var b = B/2;
        var c = C;
        var d = D/2;
        var f = E/2;
        var g = F;
        var axis_A_len = Math.sqrt(2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)/((b*b-a*c)*(Math.sqrt((a-c)*(a-c)+4*b*b) - (a+c))));
        var axis_B_len = Math.sqrt(2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)/((b*b-a*c)*(-Math.sqrt((a-c)*(a-c)+4*b*b) - (a+c))));
        if (b < 1E-12) {
            var rot = (a < c) ? 0 : Math.PI/2;
        } else {
            var rot = 0.5*Math.atan(2*b/(a-c));
            rot += (a < c) ? 0 : Math.PI/2;
        }
        var path = "M"+pts[0][0].toString()+','+pts[0][1].toString();
        path += "A"+axis_A_len.toString()+','+axis_B_len.toString() + " ";
        path += rot.toString() + " ";
        path += "0,0 "+pts[pts.length-1][0].toString()+","+pts[pts.length-1][1].toString();
        self.driveSVG.paper.ellipse(Cx, Cy, axis_B_len, axis_A_len).attr({'fill-opacity':0.0, 'stroke':'purple'});
        self.driveSVG.paper.path(path).attr({"fill-opacity":0.0,
                                             "stroke":"red"});

       // var Rworld = Math.abs(Cy);
       // console.log(Cx, Cy, Rworld);
       // // For circle: B = 0, A = C;
       // // A(Px*Px+Py*Py) + DPx + EPy + F = 0 (Px=Py=0 --> F=0) Solve remaining 3 params:
       // // A(qx*qz+qy*qy) + Dqx + Eqy (+F=0) = 0
       // // -0.5D = Cx (from equation for center points, reduced, with B/A/C subs above).
       // // -0.5E = Cy
       // var D = -2*Cx;
       // var E = -2*Cy;
       // var A = -(D*qx + E*qy) / (qx*qx + qy*qy)
       // var circleMat = [[A,   0,   D/2],
       //                  [0,   A,   E/2],
       //                  [D/2, E/2, 0]];
    }
   // self.driveSVG.node.onmousemove = self.getWorldPath;
    self.driveSVG.node.onclick = self.getWorldPath;

    self.updateLineOffsets = function (event) {
        var width =$('#drive-lines').width();
        self.lineCenterOffset = 0;//-0.08*width;
        self.lineWidthOffset = 0.18*width;
    };

    self.headStops = ['back-left', 'left','forward','right','back-right'];
    self.headStopAngles = {'back-right':[-2.85, 1.35],
                           'right':[-Math.PI/2, 1.35],
                           'forward':[0.0, 1.35],
                        //   'forward':[0.0, 1.0],
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
//        console.log("R: "+r+", Theta: "+theta);
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

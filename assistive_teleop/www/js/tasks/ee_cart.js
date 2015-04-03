RFH.CartesianEEControl = function (options) {
    'use strict';
    var self = this;
    self.name = options.name || options.arm+'EECartTask';
    self.div = options.div || 'mjpeg';
    self.arm = options.arm;
    self.side = self.arm.side[0];
    self.gripper = options.gripper;
    self.posStepSizes = {'tiny': 0.025,
        'small': 0.05,
        'medium': 0.1,
        'large': 0.25};
    self.rotStepSizes = {'tiny': Math.PI/16,
        'small': Math.PI/8,
        'medium': Math.PI/6,
        'large': Math.PI/4};
    self.tfClient = options.tfClient;
    self.ros = self.tfClient.ros;
    self.eeTF = null;
    self.cameraTF = null;
    self.eeInOpMat = null;
    self.op2baseMat = null;
    self.orientRot = 0;
    self.camera = options.camera;
    self.dt = 500; //hold-repeat time in ms
    self.mode = "table" // "wall", "free"

    self.headTableCB = function (msg) {
        var ang
        switch (self.mode) {
            case 'table':
                ang = Math.PI/2 - msg.actual.positions[1]
                break;
            case 'wall':
                ang = -msg.actual.positions[1]
                break;
            case 'free':
            default:
                ang = 0;
                break;
        }
        $('#armCtrlContainer').css({'transform':'rotateX('+ang.toString()+'rad)'});
    }
    RFH.pr2.head.stateCBList.push(self.headTableCB);

    self.focusPoint = new RFH.FocalPoint({camera: self.camera,
        tfClient: self.tfClient,
        ros: self.ros,
        side: self.arm.side,
        divId: self.div,
    });

    self.pixel23d = new RFH.Pixel23DClient({
        ros: self.ros,
        cameraInfoTopic: '/head_mount_kinect/rgb_lowres/camera_info',
        serviceName: '/pixel_2_3d'
    });

    self.buttonText = self.side === 'r' ? 'Right_Hand' : 'Left_Hand';
    self.buttonClass = 'hand-button';
    $('#touchspot-toggle, #select-focus-toggle, #toward-button, #away-button, #wristCW, #wristCCW').button();
    $('#speedOptions-buttons, #posrot-set, #ee-mode-set').buttonset();
    $('#touchspot-toggle, #touchspot-toggle-label, #select-focus-toggle, #select-focus-toggle-label, #toward-button, #away-button, #armCtrlContainer, #wristCW, #wristCCW').hide();
    $('#ctrl-ring .center').on('mousedown.rfh', function (e) { e.stopPropagation() });

    self.getStepSize = function () {
        return $('input[name=speedOption]:checked').attr('id');
    };

    self.orientHand = function () {
        if (self.focusPoint.point === null) {
            throw "Orient Hand: No focus point.";
        } 
        var target = self.focusPoint.point.clone(); // 3D point in /base_link to point at
        var eePos =  self.eeTF.translation.clone(); // 3D point in /base_link from which to point
        var camPos = self.cameraTF.translation.clone(); // 3D point of view (resolve free rotation to orient hand second axis toward camera)
        var x = new THREE.Vector3();
        var y = new THREE.Vector3();
        var z = new THREE.Vector3();
        x.subVectors(target, eePos).normalize();
        if (x.length() === 0) {
            throw "Orient Hand: End effector and target at same position"
        };
        z.subVectors(camPos, eePos).normalize();
        if (z.length() === 0) {
            throw "Orient Hand: End effector and camera at same position"
        };
        y.crossVectors(z,x).normalize();
        if (y.length() === 0) {
            throw "Orient Hand: Gimbal-lock - Camera, End Effector, and Target aligned."
        };
        z.crossVectors(x,y).normalize();
        var rotMat = new THREE.Matrix4();
        rotMat.elements[0] = x.x; rotMat.elements[4] = y.x; rotMat.elements[8] = z.x;
        rotMat.elements[1] = x.y; rotMat.elements[5] = y.y; rotMat.elements[9] = z.y;
        rotMat.elements[2] = x.z; rotMat.elements[6] = y.z; rotMat.elements[10] = z.z;
        return new THREE.Quaternion().setFromRotationMatrix(rotMat);
    };

    self.updateOpFrame = function () {
        // Define an 'operational frame' at the end effector (ee) aligned with the perspective of the camera
        // (i.e. z-axis along line from camera center through ee center for z-out optical frames, 
        //  x- and y-axes rotated accordingly.  Commands will be issued in this frame, so they have the exact
        // effect that it looks like they should from the flat, 2D camera view at all times.

        // Check that we have values for both camera and ee frames
        if (self.eeTF === null || self.cameraTF === null) { return; };
        // Format ee frame as transformation matrix
        var eePosInBase = new THREE.Vector3().copy(self.eeTF.translation);
        var eeQuatInBase = new THREE.Quaternion(self.eeTF.rotation.x, 
                self.eeTF.rotation.y,
                self.eeTF.rotation.z,
                self.eeTF.rotation.w);
        var eeInBase = new THREE.Matrix4();
        eeInBase.makeRotationFromQuaternion(eeQuatInBase);
        eeInBase.setPosition(eePosInBase);

        // Format camera frame as transformation matrix
        var camPosInBase = new THREE.Vector3().copy(self.cameraTF.translation);
        var camQuatInBase = new THREE.Quaternion(self.cameraTF.rotation.x,
                self.cameraTF.rotation.y,
                self.cameraTF.rotation.z,
                self.cameraTF.rotation.w);
        var camInBase = new THREE.Matrix4();
        camInBase.makeRotationFromQuaternion(camQuatInBase);
        camInBase.setPosition(camPosInBase);

        // Get ee position in camera frame
        var base2Cam = new THREE.Matrix4().getInverse(camInBase);
        var eePosInCam = eePosInBase.clone().applyMatrix4(base2Cam);

        // Get operational frame (op) in camera frame
        var delAngX = -Math.atan2(eePosInCam.y, eePosInCam.z);
        var delAngY = Math.atan2(eePosInCam.x, eePosInCam.z);
        var delAngEuler = new THREE.Euler(delAngX, delAngY, 0);
        var opInCamMat = new THREE.Matrix4();
        opInCamMat.makeRotationFromEuler(delAngEuler);
        opInCamMat.setPosition(eePosInCam);

        // Get EE pose in operational frame
        var eeInOpMat = new THREE.Matrix4().multiplyMatrices(base2Cam, eeInBase.clone()); // eeInOpMat is ee in cam frame
        var cam2Op = new THREE.Matrix4().getInverse(opInCamMat);
        eeInOpMat.multiplyMatrices(cam2Op, eeInOpMat); // eeInOpMat is ee in Operational Frame
        self.eeInOpMat = eeInOpMat.clone(); //Only store accessible to other functions once fully formed
        self.op2baseMat = new THREE.Matrix4().multiplyMatrices(camInBase, opInCamMat);

        // Receive Command Here, apply to current ee in op frame
        var cmdDelPos = new THREE.Vector3(0,0,0);
        var cmdDelRot = new THREE.Euler(0,0,0.5*Math.PI);
        var cmd = new THREE.Matrix4().makeRotationFromEuler(cmdDelRot);
        cmd.setPosition(cmdDelPos);
        var goalInOpMat = new THREE.Matrix4().multiplyMatrices(cmd, self.eeInOpMat.clone());
        //Transform goal back to base frame
        goalInOpMat.multiplyMatrices(self.op2baseMat, goalInOpMat);
    };

    // Get EE frame updates from TF
    if (self.arm.ee_frame !== '') {
        self.tfClient.subscribe(self.arm.ee_frame, function (tf) {
            self.eeTF = tf;
            self.updateOpFrame();
        });
        console.log("Subscribing to TF Frame: "+self.arm.ee_frame);
    } else {
        console.log("Empty EE Frame for " + self.arm.side + " arm.");
    };

    // Get camera frame updates from TF
    self.checkCameraTF = function () {
        if (self.camera.frame_id !== '') {
            self.tfClient.subscribe(self.camera.frame_id, function (tf) { 
                self.cameraTF = tf;
                self.updateOpFrame();
            }
            );
        } else {
            setTimeout(self.checkCameraTF, 500);
        }
    };
    self.checkCameraTF();

    //    self.rotX = function(poseInBaseMat, ang) {
    //        ang = (ang) ? ang : self.orientRot;
    //        var wristRotMat = new THREE.Matrix4().makeRotationX(ang);
    //        var goalRotMat = new THREE.Matrix4().extractRotation(poseInBaseMat);
    //        goalRotMat.multiply(wristRotMat);
    //        goalRotMa
    //
    //    }

    self.eeDeltaCmd = function (xyzrpy) {
        // Get default values for unspecified options
        var x = xyzrpy.x || 0.0;
        var y = xyzrpy.y || 0.0;
        var z = xyzrpy.z || 0.0;
        var roll = xyzrpy.roll || 0.0;
        var pitch = xyzrpy.pitch || 0.0;
        var yaw = xyzrpy.yaw || 0.0;
        var posStep = self.posStepSizes[self.getStepSize()]
        var rotStep = self.rotStepSizes[self.getStepSize()]

        switch (self.mode) {
            case 'table':
                if (self.eeTF === null) {
                    console.warn("Hand Data not available to send commands.");
                    return;
                }
                var handAng = Math.atan2(self.eeTF.translation.y, self.eeTF.translation.x);
                var clickAng = Math.atan2(y,x) - Math.PI/2;
                var goalAng = handAng + clickAng;
                var dx = (x === 0.0) ? 0.0 : posStep * Math.cos(goalAng);
                var dy = (y === 0.0) ? 0.0 : posStep * Math.sin(goalAng);
                var dz = posStep * z;

                var frame = self.tfClient.fixedFrame;
                
                var pos = {x: self.eeTF.translation.x + dx,
                           y: self.eeTF.translation.y + dy,
                           z: self.eeTF.translation.z - dz}
                var quat = {x: self.eeTF.rotation.x,
                        y: self.eeTF.rotation.y,
                        z: self.eeTF.rotation.z,
                        w: self.eeTF.rotation.w}
                break;
            case 'wall':
                //TODO: Check this out, it's not quite right yet!
                var handAng = Math.atan2(self.eeTF.translation.y, self.eeTF.translation.x);
                var clickAng = Math.atan2(y,x) - Math.PI/2;
                var goalAng = handAng + clickAng;
                var dx = (x === 0.0) ? 0.0 : posStep * Math.cos(goalAng);
                var dy = (y === 0.0) ? 0.0 : posStep * Math.sin(goalAng);
                var dz = posStep * z;

                var frame = self.tfClient.fixedFrame;
                var pos = {x: self.eeTF.translation.x + dz,
                           y: self.eeTF.translation.y + dy,
                           z: self.eeTF.translation.z + dx}
                var quat = {x: self.eeTF.rotation.x,
                        y: self.eeTF.rotation.y,
                        z: self.eeTF.rotation.z,
                        w: self.eeTF.rotation.w}
                break;
            case 'free':
                if (self.op2baseMat === null || self.eeInOpMat === null) {
                    console.warn("Hand Data not available to send commands.");
                    return;
                };
                // Convert to Matrix4
                var cmdDelPos = new THREE.Vector3(posStep*x, -posStep*y, posStep*z);
                var cmdDelRot = new THREE.Euler(-rotStep*roll, -rotStep*pitch, rotStep*yaw);
                var cmd = new THREE.Matrix4().makeRotationFromEuler(cmdDelRot);
                cmd.setPosition(cmdDelPos);
                // Apply delta to current ee position
                var goalInOpMat = new THREE.Matrix4().multiplyMatrices(cmd, self.eeInOpMat.clone());
                //Transform goal back to base frame
                var goalInBaseMat = new THREE.Matrix4().multiplyMatrices(self.op2baseMat, goalInOpMat);
                // Compose and send ros msg
                var pos = new THREE.Vector3();
                var quat = new THREE.Quaternion();
                var scale = new THREE.Vector3();
                goalInBaseMat.decompose(pos, quat, scale);
                try {
                    quat = self.orientHand();
                }
                catch (e) {
                    console.log(e); // log error and keep moving
                }
                var frame = self.tfClient.fixedFrame;
                break;
            default:
                console.warn("Unknown arm control mode.");
                return;
        } // End mode switch-case

        quat = new ROSLIB.Quaternion({x:quat.x, y:quat.y, z:quat.z, w:quat.w});
        self.arm.sendGoal({position: pos,
            orientation: quat,
            frame_id: frame});
    };

    self.checkMouseButtonDecorator = function (f) {
        return function (e) {
            if (e.which === 1) {
                f(e);
            } else {
                e.stopPropagation();
                return;
            }
        }
    };
     
    self.ctrlRingActivate = self.checkMouseButtonDecorator(function (e) {
        $('#ctrl-ring, #ctrl-ring > .arrow').removeClass('default').addClass('active');
        var pt = RFH.positionInElement(e);
        var w = $(e.target).width();
        var h = $(e.target).height();
        var ang = Math.atan2(-(pt[1]-h/2)/h, (pt[0]-w/2)/w);
        var delX = Math.cos(ang);
        var delY = Math.sin(ang)
        var ringMove = function (dX, dY) {
            if ( !$('#ctrl-ring').hasClass('active') ) { return; }
            self.eeDeltaCmd({x: dX, y: dY});
            setTimeout(function() {ringMove(dX, dY);} , self.dt);
        }
        ringMove(delX, delY);
    });

    self.ctrlRingActivateRot = self.checkMouseButtonDecorator(function (e) {
        $('#ctrl-ring').removeClass('default').addClass('active');
        var pt = RFH.positionInElement(e);
        var w = $(e.target).width();
        var h = $(e.target).height();
        var delX = (pt[0]-w/2)/w;
        var delY = -(pt[1]-w/2)/h;
        var ringMove = function (dX, dY) {
            if ( !$('#ctrl-ring').hasClass('active') ){ return; }
            self.eeDeltaCmd({roll: delY, pitch: delX});
            setTimeout(function() {ringMove(dX, dY);} , self.dt);
        }
        ringMove(delX, delY);
    });

    self.Inactivate = function (e) {
        $(e.target).removeClass('active').addClass('default');
    };

    self.awayCB = self.checkMouseButtonDecorator(function (e) {
        $(e.target).removeClass('default').addClass('active');
        var moveCB = function() {
            if ( !$('#away-button').hasClass('active') ) {return};
            self.eeDeltaCmd({z: 1});
            setTimeout(moveCB, self.dt);
        }
        moveCB();
    });

    self.towardCB = self.checkMouseButtonDecorator(function (e) {
        $(e.target).removeClass('default').addClass('active');
        var moveCB = function() {
            if ( !$('#toward-button').hasClass('active') ) {return};
            self.eeDeltaCmd({z: -1});
            setTimeout(moveCB, self.dt);
        }
        moveCB();
    });

    self.cwCB = self.checkMouseButtonDecorator(function (e) {
        $(e.target).removeClass('default').addClass('active');
        var moveCB = function() {
            if ( !$('#away-button').hasClass('active') ) {return};
            self.eeDeltaCmd({yaw: 1});
            setTimeout(moveCB, self.dt);
        }
        moveCB();
    });

    self.ccwCB = self.checkMouseButtonDecorator(function (e) {
        $(e.target).removeClass('default').addClass('active');
        var moveCB = function() {
            if ( !$('#toward-button').hasClass('active') ) {return};
            self.eeDeltaCmd({yaw: -1});
            setTimeout(moveCB, self.dt);
        }
        moveCB();
    });

    /// TRACK HAND WITH CAMERA ///
    self.trackHand = function () {
        clearInterval(RFH.pr2.head.pubInterval);
        RFH.pr2.head.pointHead(0, 0, 0, self.side+'_gripper_tool_frame'); // Start now, don't wait for first CB
        RFH.pr2.head.pubInterval = setInterval(function () {
            RFH.pr2.head.pointHead(0, 0, 0, self.side+'_gripper_tool_frame');
        }, 500);
    }

    /// GRIPPER SLIDER CONTROLS ///
    self.gripperDisplayDiv = self.side+'GripperDisplay';
    self.gripperDisplay = new RFH.GripperDisplay({gripper: self.gripper,
        parentId: self.div,
        divId: self.gripperDisplayDiv});
    var gripperCSS = {position: "absolute",
        height: "5%",
        width: "27%",
        bottom: "5%"};
    gripperCSS[self.arm.side] = "2%";
    $('#'+self.gripperDisplayDiv).css( gripperCSS ).hide();

    /// SELECT FOCUS POINT CONTROLS ///
    self.selectFocusCB = function (e, ui) {
        if ($('#select-focus-toggle').prop('checked')) {
            self.focusPoint.clear();
            if (self.focusPoint.point === null) {
                $('#armCtrlContainer').show();
            }
        } else {
            $('#armCtrlContainer').hide();
            var cb = function () {
                $('#armCtrlContainer').show();
                $('#select-focus-toggle').prop('checked', false).button('refresh');
                self.eeDeltaCmd({}); // Send command at current position to reorient arm
            };
            self.focusPoint.getNewFocusPoint(cb); // Pass in callback to perform cleanup/reversal
        }
    };

    self.wristCWCB = function (e) {
        self.orientRot += Math.Pi/12;
        self.orientRot = self.orientRot % 2*Math.PI;
        self.eeDeltaCmd({});
    };

    self.wristCCWCB = function (e) {
        self.orientRot -= Math.Pi/12;
        self.orientRot = self.orientRot % 2*Math.PI;
        self.eeDeltaCmd({});
    };

    self.touchSpotCB = function (e) {
        if ($('#touchspot-toggle').prop('checked')) {
            $('#armCtrlContainer').show();
            $('#'+self.div).off('click.rfh');
        } else {
            $('#armCtrlContainer').hide();
            // TODO: Change cursor here?

            var onRetCB = function (pose) {
                var quat = new THREE.Quaternion(pose.orientation.x,
                                             pose.orientation.y,
                                             pose.orientation.z,
                                             pose.orientation.w);
                var poseRotMat = new THREE.Matrix4().makeRotationFromQuaternion(quat);
                var offset = new THREE.Vector3(0.03, 0, 0) //Get to 10cm from point along normal
                offset.applyMatrix4(poseRotMat);
                var desRotMat = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(0, Math.PI, 0));
                poseRotMat.multiply(desRotMat);
                poseRotMat.setPosition(new THREE.Vector3(pose.position.x + offset.x,
                                                         pose.position.y + offset.y,
                                                         pose.position.z + offset.z));
                var trans = new THREE.Matrix4(); 
                var scale = new THREE.Vector3();
                poseRotMat.decompose(trans, quat, scale);
                self.arm.sendGoal({
                    position: new ROSLIB.Vector3({x:trans.x, y:trans.y, z:trans.z}),
                    orientation: new ROSLIB.Quaternion({x:quat.x, y:quat.y, z:quat.z, w:quat.w}),
                    frame_id: 'base_link'
                })
                $('#touchspot-toggle').prop('checked', false).button('refresh');
                $('#armCtrlContainer').show();
            };

            var clickCB = function (e) {
                var pt = RFH.positionInElement(e);
                var x = (pt[0]/e.target.width);
                var y = (pt[1]/e.target.height);
                self.pixel23d.callRelativeScale(x, y, onRetCB);
            };
            $('#'+self.div).one('click.rfh', clickCB);
        }
    };

    self.setRotationCtrls = function (e) {
        $('#ctrl-ring').off('mousedown.rfh');
        $('#toward-button, #away-button').off('click.rfh');
        self.focusPoint.clear();

        $('#ctrl-ring, #away-button, #toward-button').on('mouseup.rfh mouseout.rfh mouseleave.rfh blur.rfh', self.Inactivate)
        $('#ctrl-ring').on('mousedown.rfh', self.ctrlRingActivateRot);
        $('#away-button').on('mousedown.rfh', self.cwCB).text('CW');
        $('#toward-button').on('mousedown.rfh', self.ccwCB).text('CCW');
        $('#select-focus-toggle-label').off('click.rfh').hide();
    };

    self.setPositionCtrls = function (e) {
        $('#ctrl-ring').off('mousedown.rfh');
        $('#toward-button, #away-button').off('click.rfh');

        $('#ctrl-ring, #away-button, #toward-button').on('mouseup.rfh mouseout.rfh mouseleave.rfh blur.rfh', self.Inactivate)
        $('#ctrl-ring').on('mousedown.rfh', self.ctrlRingActivate);
        $('#away-button').on('mousedown.rfh', self.awayCB).text('');
        $('#toward-button').on('mousedown.rfh', self.towardCB).text('');
        $('#select-focus-toggle-label').on('click.rfh', self.selectFocusCB).show();
    };

    self.setEEMode = function (e) {
        self.mode = e.target.id.split("-")[2]; // Will break with different naming convention
    };

    /// TASK START/STOP ROUTINES ///
    self.start = function () {
        self.trackHand();
        $('#armCtrlContainer, #away-button, #toward-button').show();
        $("#select-focus-toggle-label").show();
        $('#speedOptions').show();
        $("#"+self.gripperDisplayDiv).show();
        $('#wristCW').on('click.rfh', self.wristCWCB).show();
        $('#wristCCW').on('click.rfh', self.wristCCWCB).show();
        $('#posrot-pos').on('click.rfh', self.setPositionCtrls);
        $('#posrot-rot').on('click.rfh', self.setRotationCtrls);
        $('#posrot-set').show();
        $('#ee-mode-set input').on('click.rfh', self.setEEMode);
        $('#ee-mode-set').show();
        $('#touchspot-toggle-label').on('click.rfh', self.touchSpotCB).show();
        $('#posrot-pos').click();
    };

    self.stop = function () {
        clearInterval(RFH.pr2.head.pubInterval);
        $('#armCtrlContainer').hide();
        $('#away-button, #toward-button').off('click.rfh').hide();
        $('#ctrl-ring').off('mouseup.rfh mouseout.rfh mouseleave.rfh blur.rfh mousedown.rfh');
        if ($('#select-focus-toggle').prop('checked')) {
            $('#select-focus-toggle').click();
        }
        $("#select-focus-toggle-label").off('click.rfh').hide();
        $('#speedOptions').hide();
        $("#"+self.gripperDisplayDiv).hide();
        $('#wristCW, #wristCCW').off('click.rfh').hide();
        $('#posrot-pos, #posrot-rot').off('click.rfh').hide();
        $('#posrot-set').hide();
        if ($('#touchspot-toggle').prop('checked')) {
            $('#touchspot-toggle').click();
        }
        $('#touchspot-toggle-label').off('click.rfh').hide();
    };
}

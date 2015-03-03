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
   $('#speedOptions-buttons, #posrot-set').buttonset();
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
        if (self.op2baseMat === null || self.eeInOpMat === null) {
                console.log("Hand Data not available to send commands.");
                return;
                };
        // Get default values for unspecified options
        var x = xyzrpy.x || 0.0;
        var y = xyzrpy.y || 0.0;
        var z = xyzrpy.z || 0.0;
        var roll = xyzrpy.roll || 0.0;
        var pitch = xyzrpy.pitch || 0.0;
        var yaw = xyzrpy.yaw || 0.0;
        // Convert to Matrix4
        var cmdDelPos = new THREE.Vector3(x, y, z);
        var cmdDelRot = new THREE.Euler(roll, pitch, yaw);
        var cmd = new THREE.Matrix4().makeRotationFromEuler(cmdDelRot);
        cmd.setPosition(cmdDelPos);
        // Apply delta to current ee position
        var goalInOpMat = new THREE.Matrix4().multiplyMatrices(cmd, self.eeInOpMat.clone());
        //Transform goal back to base frame
        var goalInBaseMat = new THREE.Matrix4().multiplyMatrices(self.op2baseMat, goalInOpMat);
        // Compose and send ros msg
        var p = new THREE.Vector3();
        var q = new THREE.Quaternion();
        var s = new THREE.Vector3();
        goalInBaseMat.decompose(p,q,s);
        try {
            q = self.orientHand();
        }
        catch (e) {
            console.log(e); // log error and keep moving
        }
        q = new ROSLIB.Quaternion({x:q.x, y:q.y, z:q.z, w:q.w});
        self.arm.sendGoal({position: p,
                           orientation: q,
                           frame_id: self.tfClient.fixedFrame});
    };

    /// POSITION CONTROLS ///
    //self.posCtrlId = self.side+'posCtrlIcon';
    //self.targetIcon = new RFH.EECartControlIcon({divId: self.posCtrlId,
    //                                             parentId: self.div,
    //                                             arm: self.arm});
    //var handCtrlCSS = {bottom:"6%"};
    //handCtrlCSS[self.arm.side] = "7%";
    //$('#'+self.posCtrlId).css(handCtrlCSS).hide();

    ///// ROTATION CONTROLS ///
    //self.rotCtrlId = self.side+'rotCtrlIcon';
    //self.rotIcon = new RFH.EERotControlIcon({divId: self.rotCtrlId,
    //                                            parentId: self.div,
    //                                             arm: self.arm});
    //$('#'+self.rotCtrlId).css(handCtrlCSS).hide();

    self.ctrlRingActivate = function (e) {
        $('#ctrl-ring, #ctrl-ring > .arrow').removeClass('default').addClass('active');
        var pt = RFH.positionInElement(e);
        var w = $(e.target).width();
        var h = $(e.target).height();
        var delX = self.posStepSizes[self.getStepSize()] * (pt[0]-w/2)/w;
        var delY = self.posStepSizes[self.getStepSize()] * (pt[1]-w/2)/h;
        self.eeDeltaCmd({x: delX, y: delY});
    };

    self.ctrlRingActivateRot = function (e) {
        $('#ctrl-ring, #ctrl-ring > .arrow').removeClass('default').addClass('active');
        var pt = RFH.positionInElement(e);
        var w = $(e.target).width();
        var h = $(e.target).height();
        var delX = self.rotStepSizes[self.getStepSize()] * (pt[0]-w/2)/w;
        var delY = self.rotStepSizes[self.getStepSize()] * (pt[1]-w/2)/h;
        self.eeDeltaCmd({roll:delY, pitch:-delX});
    };

    self.ctrlRingInactivate = function (e) {
        $('#ctrl-ring, #ctrl-ring > .arrow').removeClass('active').addClass('default');
    };

    self.awayCB = function (e) {
        self.eeDeltaCmd({z: self.posStepSizes[self.getStepSize()]});
    };

    self.towardCB = function (e) {
        self.eeDeltaCmd({z: -self.posStepSizes[self.getStepSize()]});
    };

    self.cwCB = function (e) {
        self.eeDeltaCmd({yaw: self.rotStepSizes[self.getStepSize()]});
    };

    self.ccwCB = function (e) {
        self.eeDeltaCmd({yaw: -self.rotStepSizes[self.getStepSize()]});
    };

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
        if ($('#select-focus-toggle').is(':checked')) {
            self.focusPoint.clear();
            if (self.focusPoint.point === null) {
                $('#armCtrlContainer').show();
            }
        } else {
            $('#armCtrlContainer').hide();
            var cb = function () {
                $('#armCtrlContainer').show();
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
        if ($('#touchspot-toggle').is(':checked')) {
            $('#armCtrlContainer').show();
        } else {
            $('#armCtrlContainer').hide();
            // TODO: Change cursor here?

            var onRetCB = function (pose) {
                var pt = new THREE.Vector3(pose.position);
                self.arm.sendGoal({
                    positoin: pt,
                    orientation: self.eeTF.rotation,
                    frame_id: 'base_link'
                })
                $('#touchspot-toggle').prop('checked', false);
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

        $('#ctrl-ring').on('mousedown.rfh', self.ctrlRingActivateRot);
        $('#away-button').on('click.rfh', self.cwCB).text('CW');
        $('#toward-button').on('click.rfh', self.ccwCB).text('CCW');
        $('#select-focus-toggle-label').off('click.rfh').hide();
    };

    self.setPositionCtrls = function (e) {
        $('#ctrl-ring').off('mousedown.rfh');
        $('#toward-button, #away-button, #touchspot-toggle-label').off('click.rfh');

        $('#ctrl-ring').on('mousedown.rfh', self.ctrlRingActivate);
        $('#ctrl-ring').on('mouseup.rfh mouseout.rfh mouseleave.rfh blur.rfh', self.ctrlRingInactivate)
        $('#away-button').on('click.rfh', self.awayCB).text('Away');
        $('#toward-button').on('click.rfh', self.towardCB).text('Closer');
        $('#select-focus-toggle-label').on('click.rfh', self.selectFocusCB).show();
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
        $('#touchspot-toggle-label').on('click.rfh', self.touchSpotCB).show();
        $('#posrot-pos').click();
    };
    
    self.stop = function () {
        clearInterval(RFH.pr2.head.pubInterval);
        $('#armCtrlContainer').hide();
        $('#away-button, #toward-button').off('click.rfh').hide();
        $('#ctrl-ring').off('mouseup.rfh mouseout.rfh mouseleave.rfh blur.rfh mousedown.rfh');
        if ($('#select-focus-toggle').is(':checked')) {
            $('#select-focus-toggle-label').click();
        }
        $("#select-focus-toggle-label").off('click.rfh').hide();
        $('#speedOptions').hide();
        $("#"+self.gripperDisplayDiv).hide();
        $('#wristCW, #wristCCW').off('click.rfh').hide();
        $('#posrot-pos, #posrot-rot').off('click.rfh').hide();
        $('#posrot-set').hide();
        $('#touchspot-toggle-label').off('click.rfh').hide();
    };
}
/*
RFH.EECartControlIcon = function (options) {
    'use strict';
    var self = this;
    self.divId = options.divId;
    self.parentId = options.parentId;
    self.arm = options.arm;
    self.lastDragTime = new Date();
    self.container = $('<div/>', {id: self.divId,
                                  class: "cart-ctrl-container"}).appendTo('#'+self.parentId);
    self.away = $('<div/>', {class: "away-button"}).appendTo('#'+self.divId).button();
//    self.target = $('<div/>', {class: "target-trans"}).appendTo('#'+self.divId);
    self.toward = $('<div/>', {class: "toward-button"}).appendTo('#'+self.divId).button();
    self.left = $('<div/>', {class: "left-button"}).appendTo('#'+self.divId).button().html("<");
    self.right = $('<div/>', {class: "right-button"}).appendTo('#'+self.divId).button().html(">");
    self.up = $('<div/>', {class: "up-button"}).appendTo('#'+self.divId).button().html("^");
    self.down = $('<div/>', {class: "down-button"}).appendTo('#'+self.divId).button().html("v");
//    $('#'+self.divId+' .target-trans').draggable({containment:"parent",
//                                 distance: 8,
//                                 revertDuration: 100,
//                                 revert: true})
//                                 .on("dragstart", function (event) { event.stopPropagation() });
//

    self.leftCB = function (event) {
        if ($('#'+self.divId+' .left-button').hasClass('ui-state-active')) {
            self.arm.eeDeltaCmd({x:-self.arm.stepSize});
            setTimeout(function () {self.leftCB(event)}, self.arm.dt);
        } 
    }
    $('#'+self.divId+' .left-button').on('mousedown.rfh', self.leftCB);

    self.rightCB = function (event) {
        if ($('#'+self.divId+' .right-button').hasClass('ui-state-active')) {
            self.arm.eeDeltaCmd({x:self.arm.stepSize});
            setTimeout(function () {self.rightCB(event)}, self.arm.dt);
        } 
    }
    $('#'+self.divId+' .right-button').on('mousedown.rfh', self.rightCB);

    self.upCB = function (event) {
        if ($('#'+self.divId+' .up-button').hasClass('ui-state-active')) {
            self.arm.eeDeltaCmd({y:-self.arm.stepSize});
            setTimeout(function () {self.upCB(event)}, self.arm.dt);
        } 
    }
    $('#'+self.divId+' .up-button').on('mousedown.rfh', self.upCB);

    self.downCB = function (event) {
        if ($('#'+self.divId+' .down-button').hasClass('ui-state-active')) {
            self.arm.eeDeltaCmd({y:self.arm.stepSize});
            setTimeout(function () {self.downCB(event)}, self.arm.dt);
        } 
    }
    $('#'+self.divId+' .down-button').on('mousedown.rfh', self.downCB);

    self.awayCB = function (event) {
        if ($('#'+self.divId+' .away-button').hasClass('ui-state-active')) {
            self.arm.eeDeltaCmd({z:self.arm.stepSize});
            setTimeout(function () {self.awayCB(event)}, self.arm.dt);
        } 
    }
    $('#'+self.divId+' .away-button').on('mousedown.rfh', self.awayCB);

    self.towardCB = function (event) {
        if ($('#'+self.divId+' .toward-button').hasClass('ui-state-active')){
            self.arm.eeDeltaCmd({z:-self.arm.stepSize});
            setTimeout(function () {self.towardCB(event)}, self.arm.dt);
        }
    }
    $('#'+self.divId+' .toward-button').on('mousedown.rfh', self.towardCB);

//    self.onDrag = function (event, ui) {
//        clearTimeout(self.dragTimer);
//        var time = new Date();
//        var timeleft = time - self.lastDragTime;
//        if (timeleft > 100) {
//            self.lastDragTime = time;
//            var delX = self.arm.stepSize/30 * (ui.position.left - ui.originalPosition.left);
//            var delY = self.arm.stepSize/30 * (ui.position.top - ui.originalPosition.top);
//            self.arm.eeDeltaCmd({x: delX, y: delY});
//            self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, self.arm.dt);
//        } else {
//            self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, timeleft);
//        }
//
//    }
//
//    self.dragStop = function (event, ui) {
//        clearTimeout(self.dragTimer);
//    }
//    $('#'+self.divId+' .target-trans').on('drag', self.onDrag).on('dragstop', self.dragStop);
}

RFH.EERotControlIcon = function (options) {
    'use strict';
    var self = this;
    self.divId = options.divId;
    self.parentId = options.parentId;
    self.arm = options.arm;
    self.lastDragTime = new Date();
    self.container = $('<div/>', {id: self.divId,
                                  class: "cart-ctrl-container"}).appendTo('#'+self.parentId);
    self.cwRot = $('<div/>', {class: "cw-button"}).appendTo('#'+self.divId).button();
    self.rotRight = $('<div/>', {class: "rotRight-button"}).appendTo('#'+self.divId).button().html('>');
    self.rotLeft = $('<div/>', {class: "rotLeft-button"}).appendTo('#'+self.divId).button().html('<');
    self.rotUp = $('<div/>', {class: "rotUp-button"}).appendTo('#'+self.divId).button().html('^');
    self.rotDown = $('<div/>', {class: "rotDown-button"}).appendTo('#'+self.divId).button().html('v');
//    self.target = $('<div/>', {class: "target-rot"}).appendTo('#'+self.divId);
    self.ccwRot = $('<div/>', {class: "ccw-button"}).appendTo('#'+self.divId).button();
    $('#'+self.divId+' .target-rot').on('dragstart', function(event) { event.stopPropagation()})
                                    .draggable({containment:"parent",
                                                distance: 8,
                                                revertDuration: 100,
                                                revert: true});
    self.rpy_to_quat = function (roll, pitch, yaw) {
        // Convert from RPY
        var phi = roll / 2.0;
        var the = pitch / 2.0;
        var psi = yaw / 2.0;
        var x = Math.sin(phi) * Math.cos(the) * Math.cos(psi) - 
                Math.cos(phi) * Math.sin(the) * Math.sin(psi);
        var y = Math.cos(phi) * Math.sin(the) * Math.cos(psi) + 
                Math.sin(phi) * Math.cos(the) * Math.sin(psi);
        var z = Math.cos(phi) * Math.cos(the) * Math.sin(psi) - 
                Math.sin(phi) * Math.sin(the) * Math.cos(psi);
        var w = Math.cos(phi) * Math.cos(the) * Math.cos(psi) + 
                Math.sin(phi) * Math.sin(the) * Math.sin(psi);
        var quaternion = new ROSLIB.Quaternion({x:x, y:y, z:z, w:w});
        quaternion.normalize();
        return quaternion;
        }

    self.ccwCB = function (event) {
        if ($('#'+self.divId+' .ccw-button').hasClass('ui-state-active')) {
            self.arm.eeDeltaCmd({yaw: -self.arm.dRot});
            setTimeout(function () {self.ccwCB(event)}, self.arm.dt);
        } 
    }
    $('#'+self.divId+' .ccw-button').on('mousedown.rfh', self.ccwCB);

    self.cwCB = function (event) {
        if ($('#'+self.divId+' .cw-button').hasClass('ui-state-active')){
            self.arm.eeDeltaCmd({yaw: self.arm.dRot});
            setTimeout(function () {self.cwCB(event)}, self.arm.dt);
        }
    }
    $('#'+self.divId+' .cw-button').on('mousedown.rfh', self.cwCB);

    self.rotRightCB = function (event) {
        if ($('#'+self.divId+' .rotRight-button').hasClass('ui-state-active')){
            self.arm.eeDeltaCmd({pitch: -self.arm.dRot});
            setTimeout(function () {self.rotRightCB(event)}, self.arm.dt);
        }
    }
    $('#'+self.divId+' .rotRight-button').on('mousedown.rfh', self.rotRightCB);

    self.rotLeftCB = function (event) {
        if ($('#'+self.divId+' .rotLeft-button').hasClass('ui-state-active')){
            self.arm.eeDeltaCmd({pitch: self.arm.dRot});
            setTimeout(function () {self.rotLeftCB(event)}, self.arm.dt);
        }
    }
    $('#'+self.divId+' .rotLeft-button').on('mousedown.rfh', self.rotLeftCB);

    self.rotUpCB = function (event) {
        if ($('#'+self.divId+' .rotUp-button').hasClass('ui-state-active')){
            self.arm.eeDeltaCmd({roll: -self.arm.dRot});
            setTimeout(function () {self.rotUpCB(event)}, self.arm.dt);
        }
    }
    $('#'+self.divId+' .rotUp-button').on('mousedown.rfh', self.rotUpCB);

    self.rotDownCB = function (event) {
        if ($('#'+self.divId+' .rotDown-button').hasClass('ui-state-active')){
            self.arm.eeDeltaCmd({roll: self.arm.dRot});
            setTimeout(function () {self.rotDownCB(event)}, self.arm.dt);
        }
    }
    $('#'+self.divId+' .rotDown-button').on('mousedown.rfh', self.rotDownCB);

    //self.onDrag = function (event, ui) {
    //    // x -> rot around Z
    //    // y -> rot around y
    //    clearTimeout(self.dragTimer);
    //    var time = new Date();
    //    var timeleft = time - self.lastDragTime;
    //    if (timeleft > 1000) {
    //        self.lastDragTime = time;
    //        var stepSize = self.arm.dRot * (ui.position.left - ui.originalPosition.left);
    //        var dy = self.arm.dRot * (ui.position.top - ui.originalPosition.top);
    //        self.arm.eeDeltaCmd({pitch: -stepSize, roll: dy});
    //        self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, self.arm.dt);
    //    } else {
    //        self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, timeleft);
    //    }

    //}

    //self.dragStop = function (event, ui) {
    //    clearTimeout(self.dragTimer);
    //}
    //$('#'+self.divId+' .target-rot').on('drag', self.onDrag).on('dragstop', self.dragStop);
}
*/

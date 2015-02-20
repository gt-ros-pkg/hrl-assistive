RFH.CartesianEEControl = function (options) {
    'use strict';
    var self = this;
    self.div = options.div || 'mjpeg';
    self.arm = options.arm;
    self.side = self.arm.side[0];
    self.gripper = options.gripper;
    self.arm.dx = self.arm instanceof PR2ArmJTTask ? 0.03 : 0.05;
    self.arm.dRot = self.arm instanceof PR2ArmJTTask ? Math.PI/30 : Math.PI/8; 
    self.arm.dt = self.arm instanceof PR2ArmJTTask ? 100 : 500;
    self.tfClient = options.tfClient;
    self.ros = self.tfClient.ros;
    self.eeTF = null;
    self.cameraTF = null;
    self.eeInOpMat = null;
    self.op2baseMat = null;
    self.focusPoint = new THREE.Vector3(1,0,1);
    self.camera = options.camera;
    self.buttonText = self.side === 'r' ? 'Right_Hand' : 'Left_Hand';
    self.buttonClass = 'hand-button';
    $('#touchspot-toggle').button()
    $('#touchspot-toggle-label').hide();

    self.orientHand = function (pos) {
        var target = self.focusPoint.clone(); // 3D point in /base_link to point at
        var eePos = pos !== null ? pos : self.eeTF.translation.clone(); // 3D point in /base_link from which to point
        var camPos = self.cameraTF.translation.clone(); // 3D point of view (resolve free rotation to orient hand second axis toward camera)
        var x = new THREE.Vector3();
        var y = new THREE.Vector3();
        var z = new THREE.Vector3();
        x.subVectors(target, eePos).normalize();
        z.subVectors(camPos, eePos).normalize();
        y.crossVectors(z,x).normalize();

        var rotMat = new THREE.Matrix4();
        rotMat[0] = x.x; rotMat[4] = y.x; rotMat[8] = z.x;
        rotMat[1] = x.y; rotMat[5] = y.y; rotMat[9] = z.y;
        rotMat[2] = x.z; rotMat[6] = y.z; rotMat[10] = z.z;
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

    self.arm.eeDeltaCmd = function (xyzrpy) {
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
        goalInOpMat.multiplyMatrices(self.op2baseMat, goalInOpMat);
        // Compose and send ros msg
        var p = new THREE.Vector3();
        var q = new THREE.Quaternion();
        var s = new THREE.Vector3();
        goalInOpMat.decompose(p,q,s);
        q = self.orientHand(p);
        q = new ROSLIB.Quaternion({x:q.x, y:q.y, z:q.z, w:q.w});
        self.arm.sendGoal({position: p,
                           orientation: q,
                           frame_id: self.tfClient.fixedFrame});
    };


    /// POSITION CONTROLS ///
    self.posCtrlId = self.side+'posCtrlIcon';
    self.targetIcon = new RFH.EECartControlIcon({divId: self.posCtrlId,
                                                 parentId: self.div,
                                                 arm: self.arm});
    var handCtrlCSS = {bottom:"6%"};
    handCtrlCSS[self.arm.side] = "7%";
    $('#'+self.posCtrlId).css(handCtrlCSS).hide();

    /// ROTATION CONTROLS ///
    self.rotCtrlId = self.side+'rotCtrlIcon';
    self.rotIcon = new RFH.EERotControlIcon({divId: self.rotCtrlId,
                                                parentId: self.div,
                                                 arm: self.arm});
    $('#'+self.rotCtrlId).css(handCtrlCSS).hide();

    /// SWITCH POSITION AND ROTATION ///
    $('#'+self.side+'-posrot-set').buttonset().hide().on('change.rfh', function (event, ui) {
            var mode = event.target.id.slice(-3);
            $('#'+self.side+'posCtrlIcon, #'+self.side+'rotCtrlIcon').hide();
            $('#'+self.side+mode+'CtrlIcon').show();
        });

    /// TRACKING HAND WITH CAMERA ///
    self.updateTrackHand = function (event) {
        if ( $("#"+self.side+"-track-hand-toggle").is(":checked") ){
            self.trackHand();
        } else {
            clearInterval(RFH.pr2.head.pubInterval);
        }
    }

    self.trackHand = function () {
        clearInterval(RFH.pr2.head.pubInterval);
        RFH.pr2.head.pubInterval = setInterval(function () {
            RFH.pr2.head.pointHead(0, 0, 0, self.side+'_gripper_tool_frame');
        }, 100);
    }
    $("#"+self.side+"-track-hand-toggle").button().on('change.rfh', self.updateTrackHand);
    $("#"+self.side+"-track-hand-toggle-label").hide();

    /// Touch Spot Controls ///
    


    /// GRIPPER SLIDER CONTROLS ///
    self.gripperDisplayDiv = self.side+'GripperDisplay';
    self.gripperDisplay = new RFH.GripperDisplay({gripper: self.gripper,
                                                   parentId: self.div,
                                                   divId: self.gripperDisplayDiv});
    var gripperCSS = {position: "absolute",
                      height: "3%",
                      width: "25%",
                      bottom: "2%"};
    gripperCSS[self.arm.side] = "3%";
    $('#'+self.gripperDisplayDiv).css( gripperCSS ).hide();


    /// TASK START/STOP ROUTINES ///
    self.start = function () {
        $("#touchspot-toggle-label, #"+self.side+"-track-hand-toggle-label, #"+self.side+"-posrot-set").show();
        var mode = $('#'+self.side+'-posrot-set>input:checked').attr('id').slice(-3);
        $('#'+self.side+mode+'CtrlIcon').show();
        $("#"+self.gripperDisplayDiv).show();
        self.updateTrackHand();
    };
    
    self.stop = function () {
        $('#'+self.posCtrlId + ', #'+self.rotCtrlId+', #touchspot-toggle-label, #'+self.side+'-track-hand-toggle-label, #'+self.side+'-posrot-set').hide();
        clearInterval(RFH.pr2.head.pubInterval);
        $('#'+self.gripperDisplayDiv).hide();
    };
}

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
    self.target = $('<div/>', {class: "target-trans"}).appendTo('#'+self.divId);
    self.toward = $('<div/>', {class: "toward-button"}).appendTo('#'+self.divId).button();
    $('#'+self.divId+' .target-trans').draggable({containment:"parent",
                                 distance: 8,
                                 revertDuration: 100,
                                 revert: true})
                                 .on("dragstart", function (event) { event.stopPropagation() });

    self.awayCB = function (event) {
        if ($('#'+self.divId+' .away-button').hasClass('ui-state-active')) {
            self.arm.eeDeltaCmd({z:self.arm.dx});
            setTimeout(function () {self.awayCB(event)}, self.arm.dt);
        } 
    }
    $('#'+self.divId+' .away-button').on('mousedown.rfh', self.awayCB);

    self.towardCB = function (event) {
        if ($('#'+self.divId+' .toward-button').hasClass('ui-state-active')){
            self.arm.eeDeltaCmd({z:-self.arm.dx});
            setTimeout(function () {self.towardCB(event)}, self.arm.dt);
        }
    }
    $('#'+self.divId+' .toward-button').on('mousedown.rfh', self.towardCB);

    self.onDrag = function (event, ui) {
        clearTimeout(self.dragTimer);
        var time = new Date();
        var timeleft = time - self.lastDragTime;
        if (timeleft > 100) {
            self.lastDragTime = time;
            var delX = self.arm.dx/30 * (ui.position.left - ui.originalPosition.left);
            var delY = self.arm.dx/30 * (ui.position.top - ui.originalPosition.top);
            self.arm.eeDeltaCmd({x: delX, y: delY});
            self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, self.arm.dt);
        } else {
            self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, timeleft);
        }

    }

    self.dragStop = function (event, ui) {
        clearTimeout(self.dragTimer);
    }
    $('#'+self.divId+' .target-trans').on('drag', self.onDrag).on('dragstop', self.dragStop);
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
    self.target = $('<div/>', {class: "target-rot"}).appendTo('#'+self.divId);
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

    self.onDrag = function (event, ui) {
        // x -> rot around Z
        // y -> rot around y
        clearTimeout(self.dragTimer);
        var time = new Date();
        var timeleft = time - self.lastDragTime;
        if (timeleft > 1000) {
            self.lastDragTime = time;
            var dx = self.arm.dRot * (ui.position.left - ui.originalPosition.left);
            var dy = self.arm.dRot * (ui.position.top - ui.originalPosition.top);
            self.arm.eeDeltaCmd({pitch: -dx, roll: dy});
            self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, self.arm.dt);
        } else {
            self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, timeleft);
        }

    }

    self.dragStop = function (event, ui) {
        clearTimeout(self.dragTimer);
    }
    $('#'+self.divId+' .target-rot').on('drag', self.onDrag).on('dragstop', self.dragStop);
}

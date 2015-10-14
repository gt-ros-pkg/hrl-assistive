RFH.CartesianEEControl = function (options) {
    'use strict';
    var self = this;
    self.arm = options.arm;
    self.side = self.arm.side[0];
    self.name = options.name || self.side[0]+'EECartTask';
    var divId = options.div || 'video-main';
    self.$div = $('#'+divId);
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
    self.dt = 1000; //hold-repeat time in ms
    self.mode = "table"; // "wall", "free"
    self.active = false;
    self.raycaster = new THREE.Raycaster();
    self.hoveredMesh = null;
    self.clickedMesh = null;

    /// GRIPPER SLIDER CONTROLS ///
    self.gripperDisplay = new RFH.GripperDisplay({gripper: self.gripper,
        parentId: self.$div.attr('id'),
        divId: self.side +'GripperDisplay'});

    self.getMeshPointedAt = function (event) {
        var mouse = new THREE.Vector2();
        var pt = RFH.positionInElement(event);
        var canvas = RFH.viewer.renderer.getContext().canvas; 
        mouse.x = 2 * (pt[0] - canvas.width / 2) / canvas.width;
        mouse.y = -2 * (pt[1] - canvas.height / 2) / canvas.height;

        self.raycaster.setFromCamera(mouse, RFH.viewer.camera);
        var objs = self.raycaster.intersectObjects( RFH.viewer.scene.children, true );
        if (objs.length > 0 && objs[0].object.userData.side === self.side) {
            return self.rotArrows[objs[0].object.userData.direction];
        } else {
            return null;
        }
    };

    self.canvasClickCB = function (event) {
        var clickedMesh = self.getMeshPointedAt(event);
        if (clickedMesh !== null) {
            clickedMesh.cb();
        }
    };
    $('#viewer-canvas').on('click.rfh', self.canvasClickCB);

    self.canvasMousedownCB = function (event) {
        var clickedMesh = self.getMeshPointedAt(event);
        if (clickedMesh !== null) {
            clickedMesh.mesh.material.color.set(clickedMesh.mesh.userData.clickColor);
            self.clickedMesh = clickedMesh;
        }
    };
    $('#viewer-canvas').on('mousedown.rfh', self.canvasMousedownCB);

    self.canvasMouseupCB = function (event) {
        var clickedMesh = self.getMeshPointedAt(event);
        if (clickedMesh !== null) {
            clickedMesh.mesh.material.color.set(clickedMesh.mesh.userData.hoverColor);
        } else {
            if (self.clickedMesh !== null) {
                self.clickedMesh.mesh.material.color.set(self.clickedMesh.mesh.userData.defaultColor);
                self.clickedMesh = null;
            }
        }
    };
    $('#viewer-canvas').on('mouseup.rfh', self.canvasMouseupCB);

    self.canvasMouseMoveCB = function (event) {
        var overMesh = self.getMeshPointedAt(event);
        if (overMesh === null) {
            if (self.hoveredMesh !== null){
                self.hoveredMesh.mesh.material.color.set(self.hoveredMesh.mesh.userData.defaultColor);
                self.hoveredMesh = null;
            }
        } else {
            if (self.hoveredMesh === null) {
                overMesh.mesh.material.color.set(overMesh.mesh.userData.hoverColor);
                self.hoveredMesh = overMesh;
            } else if (overMesh !== self.hoveredMesh) {
                overMesh.mesh.material.color.set(overMesh.mesh.userData.hoverColor);
                self.hoveredMesh.mesh.material.color.set(self.hoveredMesh.mesh.userData.defaultColor);
                self.hoveredMesh = overMesh;
            }
        }
    };
    $('#viewer-canvas').on('mousemove.rfh', self.canvasMouseMoveCB);

    self.rotArrowLoader = new THREE.ColladaLoader();
    var arrowOnLoad = function (collada) {
        var arrowGeom = collada.scene.children[0].children[0].geometry.clone();
        var baseMaterial = new THREE.MeshLambertMaterial();
        baseMaterial.transparent = true;
        baseMaterial.opacity = 0.67;
        self.rotArrows = {};
        var scaleX = 0.00075;
        var scaleY = 0.00075;
        var scaleZ = 0.00075;
        var edgeColor = new THREE.Color(0.1,0.1,0.1);
        var edgeMinAngle = 45;

        //Create arrow meshes for each directional control
        var mesh, edges, pos, rot, mat, cb;
        // X-Positive Rotation 3D Arrow
        baseMaterial.color.setRGB(2.75,0.1,0.1); //Something funny means RGB colors are rendered on a 0-3 scale...
        mesh = new THREE.Mesh(arrowGeom.clone(), baseMaterial.clone());
        mesh.userData.direction = 'xn';
        mesh.userData.defaultColor = new THREE.Color().setRGB(2.75,0.1,0.1);
        mesh.userData.hoverColor = new THREE.Color().setRGB(3, 0.1, 0.1);
        mesh.userData.clickColor = new THREE.Color().setRGB(3, 1, 1);
        mesh.userData.side = self.side;
        mesh.scale.set(scaleX, scaleY, scaleZ);
        edges = new THREE.EdgesHelper(mesh, edgeColor, edgeMinAngle);
        pos = new THREE.Vector3(-0.1, 0.13, 0.13);
        rot = new THREE.Euler(Math.PI/2, 0, -Math.PI/2);
        mat = new THREE.Matrix4().makeRotationFromEuler(rot);
        mat.setPosition(pos);
        cb = function (event) {self.eeDeltaCmd({'roll':1});};
        self.rotArrows[mesh.userData.direction] = {'mesh': mesh, 'edges': edges, 'transform': mat, 'cb': cb};
        // X-Negative Rotation 3D Arrow
        mesh = new THREE.Mesh(arrowGeom.clone(), baseMaterial.clone());
        mesh.userData.direction = 'xp';
        mesh.userData.defaultColor = new THREE.Color().setRGB(2.75,0.1,0.1);
        mesh.userData.hoverColor = new THREE.Color().setRGB(3, 0.1, 0.1);
        mesh.userData.clickColor = new THREE.Color().setRGB(3, 1, 1);
        mesh.userData.side = self.side;
        mesh.scale.set(scaleX, scaleY, scaleZ);
        edges = new THREE.EdgesHelper(mesh, edgeColor, edgeMinAngle);
        pos = new THREE.Vector3(-0.1, -0.13, 0.13);
        rot = new THREE.Euler(-Math.PI/2, 0, Math.PI/2);
        mat = new THREE.Matrix4().makeRotationFromEuler(rot);
        mat.setPosition(pos);
        cb = function (event) {self.eeDeltaCmd({'roll':-1});};
        self.rotArrows[mesh.userData.direction] = {'mesh': mesh, 'edges': edges, 'transform': mat, 'cb': cb};
        // Y-Positive Rotation 3D Arrow
        baseMaterial.color.setRGB(0.1, 2.75, 0.1);
        mesh = new THREE.Mesh(arrowGeom.clone(), baseMaterial.clone());
        mesh.userData.direction = 'yn';
        mesh.userData.defaultColor = new THREE.Color().setRGB(0.1, 2.75, 0.1);
        mesh.userData.hoverColor = new THREE.Color().setRGB(0.1, 3, 0.1);
        mesh.userData.clickColor = new THREE.Color().setRGB(1, 3, 1);
        mesh.userData.side = self.side;
        mesh.scale.set(scaleX, scaleY, scaleZ);
        edges = new THREE.EdgesHelper(mesh, edgeColor, edgeMinAngle);
        pos = new THREE.Vector3(-0.13, -0.025, -0.13);
        rot = new THREE.Euler(0, 0, 0);
        mat = new THREE.Matrix4().makeRotationFromEuler(rot);
        mat.setPosition(pos);
        cb = function (event) {self.eeDeltaCmd({'pitch':1});};
        self.rotArrows[mesh.userData.direction] = {'mesh': mesh, 'edges': edges, 'transform': mat, 'cb': cb};
        // Y-Negative Rotation 3D Arrow
        mesh = new THREE.Mesh(arrowGeom.clone(), baseMaterial.clone());
        mesh.userData.direction = 'yp';
        mesh.userData.defaultColor = new THREE.Color().setRGB(0.1, 2.75, 0.1);
        mesh.userData.hoverColor = new THREE.Color().setRGB(0.1, 3, 0.1);
        mesh.userData.clickColor = new THREE.Color().setRGB(1, 3, 1);
        mesh.userData.side = self.side;
        mesh.scale.set(scaleX, scaleY, scaleZ);
        edges = new THREE.EdgesHelper(mesh, edgeColor, edgeMinAngle);
        pos = new THREE.Vector3(-0.13, 0.025, 0.13);
        rot = new THREE.Euler(Math.PI,0,0);
        mat = new THREE.Matrix4().makeRotationFromEuler(rot);
        mat.setPosition(pos);
        cb = function (event) {self.eeDeltaCmd({'pitch':-1});};
        self.rotArrows[mesh.userData.direction] = {'mesh': mesh, 'edges': edges, 'transform': mat, 'cb': cb};
        // Z-Positive Rotation 3D Arrow
        baseMaterial.color.setRGB(0.1,0.1,2.75);
        mesh = new THREE.Mesh(arrowGeom.clone(), baseMaterial.clone());
        mesh.userData.direction = 'zp';
        mesh.userData.defaultColor = new THREE.Color().setRGB(0.1, 0.1, 2.75);
        mesh.userData.hoverColor = new THREE.Color().setRGB(0.1, 0.1, 3);
        mesh.userData.clickColor = new THREE.Color().setRGB(1, 1, 3);
        mesh.userData.side = self.side;
        mesh.scale.set(scaleX, scaleY, scaleZ);
        edges = new THREE.EdgesHelper(mesh, edgeColor, edgeMinAngle);
        pos = new THREE.Vector3(-0.13, -0.13, 0.025);
        rot = new THREE.Euler(-Math.PI/2, 0, 0);
        mat = new THREE.Matrix4().makeRotationFromEuler(rot);
        mat.setPosition(pos);
        cb = function (event) {self.eeDeltaCmd({'yaw':1});};
        self.rotArrows[mesh.userData.direction] = {'mesh': mesh, 'edges': edges, 'transform': mat, 'cb': cb};
        // Z-Negative Rotation 3D Arrow
        mesh = new THREE.Mesh(arrowGeom.clone(), baseMaterial.clone());
        mesh.userData.direction = 'zn';
        mesh.userData.defaultColor = new THREE.Color().setRGB(0.1, 0.1, 2.75);
        mesh.userData.hoverColor = new THREE.Color().setRGB(0.1, 0.1, 3);
        mesh.userData.clickColor = new THREE.Color().setRGB(1, 1, 3);
        mesh.userData.side = self.side;
        mesh.scale.set(scaleX, scaleY, scaleZ);
        edges = new THREE.EdgesHelper(mesh, edgeColor, edgeMinAngle);
        pos = new THREE.Vector3(-0.13, 0.13, -0.025);
        rot = new THREE.Euler(Math.PI/2, 0, 0);
        mat = new THREE.Matrix4().makeRotationFromEuler(rot);
        mat.setPosition(pos);
        cb = function (event) {self.eeDeltaCmd({'yaw':-1});};
        self.rotArrows[mesh.userData.direction] = {'mesh': mesh, 'edges': edges, 'transform': mat, 'cb': cb};

        for (var dir in self.rotArrows) {
            self.rotArrows[dir].mesh.visible = false;
            self.rotArrows[dir].edges.visible = false;
            RFH.viewer.scene.add(self.rotArrows[dir].mesh);
            RFH.viewer.scene.add(self.rotArrows[dir].edges);
        }
    };

    var arrowOnProgress = function (data) {
        console.log("Loading Rotation Arrow Collada Mesh: ", data.loaded/data.total);
    };

    self.rotArrowLoader.load('./data/Curved_Arrow_Square.dae', arrowOnLoad, arrowOnProgress);


    self.updateRotImage = function () {
        if (self.eeTF === null) { return; }
        var q = new THREE.Quaternion(self.eeTF.rotation.x,
                                     self.eeTF.rotation.y,
                                     self.eeTF.rotation.z,
                                     self.eeTF.rotation.w);
        var tfMat = new THREE.Matrix4().makeRotationFromQuaternion(q);
        tfMat.setPosition(new THREE.Vector3(self.eeTF.translation.x,
                                            self.eeTF.translation.y,
                                            self.eeTF.translation.z));

        var arrowInWorldFrame = new THREE.Matrix4();
        var arrowPos = new THREE.Vector3();
        var arrowQuat = new THREE.Quaternion();
        var arrowScale = new THREE.Vector3();
        for (var dir in self.rotArrows) {
            arrowInWorldFrame.multiplyMatrices(tfMat, self.rotArrows[dir].transform);
            arrowInWorldFrame.decompose(arrowPos, arrowQuat, arrowScale);
            self.rotArrows[dir].mesh.position.set(arrowPos.x, arrowPos.y, arrowPos.z);
            self.rotArrows[dir].mesh.quaternion.set(arrowQuat.x, arrowQuat.y, arrowQuat.z, arrowQuat.w);
        }
        RFH.viewer.renderer.render(RFH.viewer.scene, RFH.viewer.camera);
    };

    self.updateCtrlRingViz = function () {
        // Check that we have values for both camera and ee frames
        if (!self.active) { return; }
        if (self.mode === 'free') {
            $('#armCtrlContainer').css({'transform':'none'});
            return;
        }
        if (self.eeTF === null || self.cameraTF === null) { 
            console.log("Cannot update hand control ring, missing tf information");
            return;
        }
        var eePos =  self.eeTF.translation.clone(); // 3D Hand position in /base_link
        var camPos = self.cameraTF.translation.clone(); // 3D camera position in /base_link
        var transformStr;

        if (self.mode !== 'free') {
            var camQuat = self.cameraTF.rotation.clone();
            camQuat = new THREE.Quaternion(camQuat.x, camQuat.y, camQuat.z, camQuat.w);
            camQuat.multiply(new THREE.Quaternion(0.5, -0.5, 0.5, 0.5));//Rotate from optical frame to link
            var camEuler = new THREE.Euler().setFromQuaternion(camQuat, 'ZYX');// NO IDEA, but it works with this order...
            var rot = camEuler.z;//Rotation around Z -- counter rotate icon to keep arrow pointed forward.
            var dx = eePos.x - camPos.x;
            var dy = eePos.y - camPos.y;
            var dz = eePos.z - camPos.z;
            var dxy = Math.sqrt(dx*dx + dy*dy);
            var phi = Math.atan2(dxy, dz) - Math.PI/2; // Angle from horizontal

            switch (self.mode) {
                case 'table':
                    var rotX = phi - Math.PI/2;
                    transformStr = "rotateX("+rotX.toString()+"rad) rotate("+rot.toString()+"rad)";
                    break;
                case 'wall':
                    transformStr = "rotateX("+phi.toString()+"rad) rotateY("+rot.toString()+"rad)";
                    break;
            }
        } else {
            transformStr = 'none';
        }
        //TODO: Clean up scaling so that it is useful.  See if it worsens visual understanding...
        var rect = $('#armCtrlContainer')[0].getBoundingClientRect();
        var videoHeight = $('#armCtrlContainer').parent().height();
        var videoWidth = $('#armCtrlContainer').parent().width();
        var ratio = Math.max(rect.height/videoHeight, rect.width/videoWidth);
        $('#armCtrlContainer').css({'transform':transformStr});
    };

    self.focusPoint = new RFH.FocalPoint({camera: self.camera,
        tfClient: self.tfClient,
        ros: self.ros,
        side: self.arm.side,
        divId: self.$div.id,
    });

    self.pixel23d = new RFH.Pixel23DClient({
        ros: self.ros,
        cameraInfoTopic: '/head_mount_kinect/rgb_lowres/camera_info',
        serviceName: '/pixel_2_3d'
    });

    self.buttonText = self.side === 'r' ? 'Right_Hand' : 'Left_Hand';
    self.buttonClass = 'hand-button';
    $('#touchspot-toggle, #select-focus-toggle, #toward-button, #away-button').button();
    $('#speedOptions-buttons, #'+self.side+'-posrot-set, #ee-mode-set').buttonset();
    $('#touchspot-toggle, #touchspot-toggle-label, #select-focus-toggle, #select-focus-toggle-label, #toward-button, #away-button, #armCtrlContainer').hide();
    $('#ctrl-ring .center').on('mousedown.rfh', function (e) { e.stopPropagation(); });

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
            throw "Orient Hand: End effector and target at same position";
        }
        z.subVectors(camPos, eePos).normalize();
        if (z.length() === 0) {
            throw "Orient Hand: End effector and camera at same position";
        }
        y.crossVectors(z,x).normalize();
        if (y.length() === 0) {
            throw "Orient Hand: Gimbal-lock - Camera, End Effector, and Target aligned.";
        }
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
        if (self.eeTF === null || self.cameraTF === null) { return; }
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
            self.updateCtrlRingViz();
            self.updateRotImage();
        });
        console.log("Subscribing to TF Frame: "+self.arm.ee_frame);
    } else {
        console.log("Empty EE Frame for " + self.arm.side + " arm.");
    }

    // Get camera frame updates from TF
    self.checkCameraTF = function () {
        if (self.camera.frame_id !== '') {
            self.tfClient.subscribe(self.camera.frame_id, function (tf) {
                self.cameraTF = tf;
                self.updateOpFrame();
                self.updateCtrlRingViz();
            }
            );
        } else {
            setTimeout(self.checkCameraTF, 500);
        }
    };
    self.checkCameraTF();

    self.eeDeltaCmd = function (xyzrpy) {
        // Get default values for unspecified options
        var x = xyzrpy.x || 0.0;
        var y = xyzrpy.y || 0.0;
        var z = xyzrpy.z || 0.0;
        var roll = xyzrpy.roll || 0.0;
        var pitch = xyzrpy.pitch || 0.0;
        var yaw = xyzrpy.yaw || 0.0;
        var posStep = self.posStepSizes[self.getStepSize()];
        var rotStep = self.rotStepSizes[self.getStepSize()];
        var handAng;
        var clickAng;
        var goalAng;
        var dx;
        var dy;
        var dz;
        var dRoll;
        var dPitch;
        var dYaw;

        switch (self.mode) {
            case 'table':
                if (self.eeTF === null) {
                    console.warn("Hand Data not available to send commands.");
                    return;
                }
                 handAng = Math.atan2(self.eeTF.translation.y, self.eeTF.translation.x);
                 clickAng = Math.atan2(y,x) - Math.PI/2;
                 goalAng = handAng + clickAng;
                 dx = (x === 0.0) ? 0.0 : posStep * Math.cos(goalAng);
                 dy = (y === 0.0) ? 0.0 : posStep * Math.sin(goalAng);
                 dz = posStep * z;
                 dRoll = rotStep * roll;
                 dPitch = rotStep * pitch;
                 dYaw = rotStep * yaw;
                // Convert del goal to Matrix4
                var cmdDelPos = new THREE.Vector3(posStep*x, -posStep*y, posStep*z);
                var cmdDelRot = new THREE.Euler(-rotStep*roll, -rotStep*pitch, rotStep*yaw);
                var cmd = new THREE.Matrix4().makeRotationFromEuler(cmdDelRot);
                cmd.setPosition(cmdDelPos);
                // Get EE transform in THREE mat4
                var eeQuat = new THREE.Quaternion(self.eeTF.rotation.x,
                                                 self.eeTF.rotation.y,
                                                 self.eeTF.rotation.z,
                                                 self.eeTF.rotation.w);
                var eeMat = new THREE.Matrix4().makeRotationFromQuaternion(eeQuat);
                // Transform del goal (in hand frame) to base frame
                cmd.multiplyMatrices(eeMat, cmd);
                var pos = new THREE.Vector3();
                var quat = new THREE.Quaternion();
                var scale = new THREE.Vector3();
                cmd.decompose(pos, quat, scale);
                pos.x += dx;
                pos.y += dy;
                pos.z += dz;
                break;
            case 'wall':
                if (self.eeTF === null) {
                    console.warn("Hand Data not available to send commands.");
                    return;
                }
                 handAng = Math.atan2(self.eeTF.translation.y, self.eeTF.translation.x);
                 clickAng = Math.atan2(y,x) - Math.PI/2;
                 goalAng = clickAng;
                 dx = posStep * z;
                 dz = (x === 0.0) ? 0.0 : -posStep * Math.cos(goalAng);
                 dy = (y === 0.0) ? 0.0 : posStep * Math.sin(goalAng);
                break;
            case 'free':
                if (self.op2baseMat === null || self.eeInOpMat === null) {
                    console.warn("Hand Data not available to send commands.");
                    return;
                }
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

        var frame = self.tfClient.fixedFrame;
        var pos = {x: self.eeTF.translation.x + dx,
                   y: self.eeTF.translation.y + dy,
                   z: self.eeTF.translation.z - dz};
        quat = new ROSLIB.Quaternion({x:quat.x, y:quat.y, z:quat.z, w:quat.w});
        self.arm.sendPoseGoal({position: pos,
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
        };
    };

    self.ctrlRingActivate = self.checkMouseButtonDecorator(function (e) {
        $('#ctrl-ring, #ctrl-ring > .arrow').removeClass('default').addClass('active');
        var pt = RFH.positionInElement(e);
        var w = $(e.target).width();
        var h = $(e.target).height();
        var ang = Math.atan2(-(pt[1]-h/2)/h, (pt[0]-w/2)/w);
        var delX = Math.cos(ang);
        var delY = Math.sin(ang);
        var ringMove = function (dX, dY) {
            if ( !$('#ctrl-ring').hasClass('active') ) { return; }
            self.eeDeltaCmd({x: dX, y: dY});
            setTimeout(function() {ringMove(dX, dY);} , self.dt);
        };
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
        };
        ringMove(delX, delY);
    });

    self.Inactivate = function (e) {
        $(e.target).removeClass('active').addClass('default');
    };

    self.awayCB = self.checkMouseButtonDecorator(function (e) {
        $(e.target).removeClass('default').addClass('active');
        var moveCB = function() {
            if ( !$('#away-button').hasClass('active') ) {return;}
            self.eeDeltaCmd({z: 1});
            setTimeout(moveCB, self.dt);
        };
        moveCB();
    });

    self.towardCB = self.checkMouseButtonDecorator(function (e) {
        $(e.target).removeClass('default').addClass('active');
        var moveCB = function() {
            if ( !$('#toward-button').hasClass('active') ) {return;}
            self.eeDeltaCmd({z: -1});
            setTimeout(moveCB, self.dt);
        };
        moveCB();
    });

    self.cwCB = self.checkMouseButtonDecorator(function (e) {
        $(e.target).removeClass('default').addClass('active');
        var moveCB = function() {
            if ( !$('#away-button').hasClass('active') ) {return;}
            self.eeDeltaCmd({yaw: 1});
            setTimeout(moveCB, self.dt);
        };
        moveCB();
    });

    self.ccwCB = self.checkMouseButtonDecorator(function (e) {
        $(e.target).removeClass('default').addClass('active');
        var moveCB = function() {
            if ( !$('#toward-button').hasClass('active') ) {return;}
            self.eeDeltaCmd({yaw: -1});
            setTimeout(moveCB, self.dt);
        };
        moveCB();
    });

    /// TRACK HAND WITH CAMERA ///
    self.trackHand = function (doTrack) {
        if (doTrack) {
            setTimeout(function(){ RFH.pr2.head.trackPoint(0, 0, 0, self.side+'_gripper_tool_frame');}, 20); // Wait 20 ms.  There seems to be a race condition in the prior goal being cancelled and this being sent, so it sometimes gets cancelled instead (maybe?).  This seems enough to fix it.
        } else {
            RFH.pr2.head.stopTracking();
        }
    };

    /// SELECT FOCUS POINT CONTROLS ///
//    self.selectFocusCB = function (e, ui) {
//        if ($('#select-focus-toggle').prop('checked')) {
//            self.focusPoint.clear();
//            if (self.focusPoint.point === null) {
//                $('#armCtrlContainer').show();
//            }
//        } else {
//            $('#armCtrlContainer').hide();
//            var cb = function () {
//                $('#armCtrlContainer').show();
//                $('#select-focus-toggle').prop('checked', false).button('refresh');
//                self.eeDeltaCmd({}); // Send command at current position to reorient arm
//            };
//            self.focusPoint.getNewFocusPoint(cb); // Pass in callback to perform cleanup/reversal
//        }
//    };
//
//    self.wristCWCB = function (e) {
//        self.orientRot += Math.Pi/12;
//        self.orientRot = self.orientRot % 2*Math.PI;
//        self.eeDeltaCmd({});
//    };
//
//    self.wristCCWCB = function (e) {
//        self.orientRot -= Math.Pi/12;
//        self.orientRot = self.orientRot % 2*Math.PI;
//        self.eeDeltaCmd({});
//    };

    self.touchSpotCB = function (e) {
        if ($('#touchspot-toggle').prop('checked')) {
            self.setPositionCtrls();
            self.$div.off('click.rfh');
        } else {
            $('#armCtrlContainer, .'+self.side+'-arm-rot-icon').hide();
            // TODO: Change cursor here?

            var onRetCB = function (pose_stamped) {
                var pose = pose_stamped.pose;
                console.log(pose);
                var quat = new THREE.Quaternion(pose.orientation.x,
                                             pose.orientation.y,
                                             pose.orientation.z,
                                             pose.orientation.w);
                var poseRotMat = new THREE.Matrix4().makeRotationFromQuaternion(quat);
                var offset = new THREE.Vector3(0.13, 0, 0); //Get to x dist from point along normal
                offset.applyMatrix4(poseRotMat);
                var desRotMat = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(0, Math.PI, 0));
                poseRotMat.multiply(desRotMat);
                poseRotMat.setPosition(new THREE.Vector3(pose.position.x + offset.x,
                                                         pose.position.y + offset.y,
                                                         pose.position.z + offset.z));
                var trans = new THREE.Matrix4();
                var scale = new THREE.Vector3();
                poseRotMat.decompose(trans, quat, scale);

                var trajectoryCB = function (traj) {
                    if (traj.joint_trajectory.points.length === 0) {
                        console.log("Empty Trajectory Received.");
                    } else {
                        console.log("Got Trajectory", traj);
                        self.arm.sendTrajectoryGoal(traj.joint_trajectory);
                    }
                };
                self.arm.planTrajectory({
                    position: new ROSLIB.Vector3({x:trans.x, y:trans.y, z:trans.z}),
                    orientation: new ROSLIB.Quaternion({x:quat.x, y:quat.y, z:quat.z, w:quat.w}),
                    frame_id: 'base_link',
                    cb: trajectoryCB
                });
                $('#touchspot-toggle').prop('checked', false).button('refresh');
                self.setPositionCtrls();
            };

            var clickCB = function (e) {
                var pt = RFH.positionInElement(e);
                var x = pt[0]/$(e.target).width();
                var y = pt[1]/$(e.target).height();
                self.pixel23d.callRelativeScale(x, y, onRetCB);
            };
            self.$div.one('click.rfh', clickCB);
        }
    };

    self.setRotationCtrls = function (e) {
        $('.'+self.side+'-arm-rot-icon, .'+self.side+'-arm-rot-icon-baseline').show();
        $('#armCtrlContainer').hide();
        $('#viewer-canvas').show();
        for (var dir in self.rotArrows) {
            self.rotArrows[dir].mesh.visible = true;
            self.rotArrows[dir].edges.visible = true;
        }
        $(window).resize(); // Trigger canvas to update size TODO: unreliable, inconsistent behavior -- Fix
    };

    self.setPositionCtrls = function (e) {
        $('.'+self.side+'-arm-rot-icon, .'+self.side+'-arm-rot-icon-baseline').hide();
        $('#viewer-canvas').hide();
        $('#armCtrlContainer').show();
        for (var dir in self.rotArrows) {
            self.rotArrows[dir].mesh.visible = false;
            self.rotArrows[dir].edges.visible = false;
        }
        $('#ctrl-ring, #away-button, #toward-button').on('mouseup.rfh mouseout.rfh mouseleave.rfh blur.rfh', self.Inactivate);
        $('#ctrl-ring').on('mousedown.rfh', self.ctrlRingActivate);
        $('#away-button').on('mousedown.rfh', self.awayCB);
        $('#toward-button').on('mousedown.rfh', self.towardCB);
    };

    self.setEEMode = function (e) {
        self.mode = e.target.id.split("-")[2]; // Will break with different naming convention
        self.updateCtrlRingViz();
    };

    $('#'+self.side+'-posrot-pos').on('click.rfh', self.setPositionCtrls);
    $('#'+self.side+'-posrot-rot').on('click.rfh', self.setRotationCtrls);

    /// TASK START/STOP ROUTINES ///
    self.start = function () {
        self.trackHand(true);
        $('.'+self.side+'-arm-ctrl, .arm-ctrl').show();
        $('#armCtrlContainer, #away-button, #toward-button').show();
        $("#select-focus-toggle-label").show();
        $('#speedOptions').show();
        self.gripperDisplay.show();
        $('#'+self.side+'-posrot-set').show();
        $('#ee-mode-set input').on('click.rfh', self.setEEMode);
        $('#ee-mode-set').show();
        $('#touchspot-toggle-label').on('click.rfh', self.touchSpotCB).show();
        $('#'+self.side+'-posrot-pos').click();
        self.active = true;
        self.updateCtrlRingViz();
    };

    self.stop = function () {
        self.trackHand(false);
        $('.'+self.side+'-arm-ctrl, .arm-ctrl').hide();
        $('#armCtrlContainer').hide();
        $('#away-button, #toward-button').off('mousedown.rfh').hide();
        $('#ctrl-ring').off('mouseup.rfh mouseout.rfh mouseleave.rfh blur.rfh mousedown.rfh');
        $('#viewer-canvas').hide();
//        if ($('#select-focus-toggle').prop('checked')) {
//            $('#select-focus-toggle').click();
//        }
//        $("#select-focus-toggle-label").off('click.rfh').hide();
        $('#speedOptions').hide();
        self.gripperDisplay.hide();
        if ($('#touchspot-toggle').prop('checked')) {
            $('#touchspot-toggle').click();
        }
        $('#touchspot-toggle-label').off('click.rfh').hide();
        self.active = false;
        for (var dir in self.rotArrows) {
            self.rotArrows[dir].mesh.visible = false;
            self.rotArrows[dir].edges.visible = false;
        }
    };
};


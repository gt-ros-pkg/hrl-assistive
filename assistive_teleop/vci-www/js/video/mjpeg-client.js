var RFH = (function (module) {
    module.MjpegClient = function (options) {
        "use strict";
        var self = this;
        options = options || {};
        self.imageTopic = options.imageTopic;
        self.$div = $('#'+options.divId);
        var imageId = options.imageId || self.$div.attr('id') + '-image';
        self.$imageDiv = $('#'+ imageId);
        self.server = "http://"+options.host+":"+options.port;
        self.activeParams = {'topic': options.imageTopic,
            'quality': options.quality || 50};

        self.cameraModel = new RFH.ROSCameraModel({ros: options.ros,
            infoTopic: options.infoTopic,
            rotated: options.rotated || false,
            tfClient: options.tfClient});

        self.refreshSize = function (resizeEvent) {
            if (!self.cameraModel.has_data) {
                self.$div.append("Waiting on camera information.");
                return false;
            }
            var camRatio = self.cameraModel.width/self.cameraModel.height;
            //var contWidth = $('body').width() - self.$div.offset().left;  // For left-aligned video-main
            var contWidth = $('body').width(); // For right-aligned video-main
            var contHeight = $('body').height() - self.$div.offset().top;
            var contRatio = contWidth/contHeight;
            if (contRatio > camRatio) {
                var width  = contHeight * camRatio;
                self.setParam('width', width);
                self.setParam('height', contHeight);
            } else if (contRatio < camRatio) {
                var height  = contWidth / camRatio;
                self.setParam('height', height);
                self.setParam('width', contWidth);
            }
            $(window).trigger('resize.rfh');
            return true;
        };
        $(window).on('resize', self.refreshSize);

        // Update the server and display to reflect the current properties
        self.update = function () {
            var srcStr = self.server + "/stream?";
            for (var param in self.activeParams)
            {
                srcStr += param + '=' + self.activeParams[param] + '&';
            }
            self.$imageDiv.attr("src", srcStr);
            console.log("Video Request: ", srcStr);
            self.$imageDiv.width(self.activeParams.width)
                .height(self.activeParams.height);
        };

        self.int_params = ['height', 'width', 'quality'];
        // Set parameter value
        self.setParam = function (param, value) {
            if (self.int_params.indexOf(param) >= 0) {
                value = Math.round(value);
            }
            self.activeParams[param] = value;
            self.update();
        };

        // Return parameter value
        self.getParam = function (param) {
            return self.activeParams[param];
        };
    };

    module.ROSCameraModel = function (options) {
        "use strict";
        var self = this;
        options = options || {};
        self.ros = options.ros;
        self.infoTopic = options.infoTopic;
        self.rotated = options.rotated || false;
        self.width = null;
        self.height = null;
        self.frame_id = '';
        self.tfClient = options.tfClient;
        self.transform = null;
        self.distortion_model = '';
        self.D = [0,0,0,0,0];
        self.K = [[1,0,0],[0,1,0],[0,0,1]];
        self.R = [[1,0,0],[0,1,0],[0,0,1]];
        self.KR = [[1,0,0],[0,1,0],[0,0,1]];
        self.KR_inv = [[1,0,0],[0,1,0],[0,0,1]];
        self.P = [[1,0,0,0],[0,1,0,0],[0,0,1,0]];
        self.has_data = false;

        self.ros.getMsgDetails('sensor_msgs/CameraInfo');

        self.cameraInfoSubscriber = new ROSLIB.Topic({
            ros: self.ros,     
            name: self.infoTopic,
            messageType: 'sensor_msgs/CameraInfo'});

        self.tryTFSubscribe = function () {
            if (self.frame_id !== '') {
                self.tfClient.subscribe(self.camera.frame_id, function (tf) { self.transform = tf;});
                console.log("Got camera data, subscribing to TF Frame: "+self.camera.frame_id);
            } else {
                console.log("No camera data -> no TF Transform");
                setTimeout(self.tryTFSubscribe, 500);
            }
        };

        self.infoMsgCB = function (infoMsg) {
            console.log("Updating camera model from " + self.infoTopic);
            self.frame_id = infoMsg.header.frame_id;
            if (self.rotated) { self.frame_id += '_rotated'; }
            self.width = infoMsg.width;
            self.height = infoMsg.height;
            self.distortion_model = infoMsg.distortion_model;
            self.D = infoMsg.D;
            self.K = [infoMsg.K.slice(0,3),
                infoMsg.K.slice(3,6), 
                infoMsg.K.slice(6,9)];

            self.R = [infoMsg.R.slice(0,3),
                infoMsg.R.slice(3,6), 
                infoMsg.R.slice(6,9)];

            self.P = [infoMsg.P.slice(0,4),
                infoMsg.P.slice(4,8), 
                infoMsg.P.slice(8,12)];

            // Not collecting data on binning, ROI
            self.KR = numeric.dot(self.K, self.R);
            self.KR_inv = numeric.inv(self.KR);
            self.has_data = true;
            if (self.frame_id !== '') {
                self.tfClient.subscribe(self.frame_id, function (tf) { self.transform = tf; });
                console.log("Subscribing to TF Frame: "+self.frame_id);
            } else {
                console.log("Camera at " + self.infoTopic + " reported empty frame id, cannot get TF data");
            }
            self.cameraInfoSubscriber.unsubscribe(); // Close subscriber to save bandwidth
        };

        self.infoSubCBList = [self.infoMsgCB];
        self.infoSubCB = function (msg) {
            for (var cb in self.infoSubCBList) {
                self.infoSubCBList[cb](msg); 
            }
        };
        self.updateCameraInfo = function () {
            // Re-subscribe to get new parameters
            self.has_data = false;
            self.cameraInfoSubscriber.subscribe(self.infoSubCB);
        };

        // back-project a pixel some distance into the real world
        // Returns a geoemtry_msgs/PointStamped msg
        self.projectPixel = function (px, py, dist) { 
            if (!self.has_data || self.transform === null) {
                console.error("Camera Model has not received camera info or frame transform");
                return;
            }
            var d = dist !== undefined ? dist : 2; 
            var pixel_hom = [[px],[py],[1]]; //Pixel value in homogeneous coordinates
            var vec = numeric.dot(self.KR_inv, pixel_hom);
            vec = numeric.transpose(vec)[0];
            var mag = numeric.norm2(vec);
            return numeric.mul(d/mag, vec);
        };

        self.checkFrameId = function (frame_id) {
            if (frame_id[0] === '/') { frame_id = frame_id.substring(1); }
            var fixedFrame = self.tfClient.fixedFrame.substring(1);
            return (frame_id !== fixedFrame && frame_id !== self.frame_id) ? false : true;
        };

        self.projectPoints = function (pts, frame_id) {
            if (!self.has_data || self.transform === null) {
                console.error("Camera Model has not received camera info or frame transform");
                return;
            }
            if (typeof frame_id === 'undefined') { // Assume camera frame if not specified
                frame_id = self.frame_id; 
            } else if (!self.checkFrameId(frame_id)) { 
                console.log("cameraModel -- Unknown Frame Id: ", frame_id);
                return;
            }
            var ptsFlat = pts.reduce(function(a,b){return a.concat(b);});
            if (frame_id === self.tfClient.fixedFrame.substring(1)) {
                var q = new THREE.Quaternion(self.transform.rotation.x,
                    self.transform.rotation.y,
                    self.transform.rotation.z,
                    self.transform.rotation.w);
                var tfMat = new THREE.Matrix4().makeRotationFromQuaternion(q);
                tfMat.setPosition(new THREE.Vector3(self.transform.translation.x,
                    self.transform.translation.y,
                    self.transform.translation.z));
                tfMat.getInverse(tfMat);
                ptsFlat = tfMat.applyToVector3Array(ptsFlat);
            }
            for (var i=0; i < ptsFlat.length/3; i++ ) {
                pts[i] = ptsFlat.slice(3*i,3*i+3).concat(1);
            }
            var pixel_hom = numeric.dot(self.P, numeric.transpose(pts));
            pixel_hom = numeric.transpose(pixel_hom);
            for (var j=0; j < pixel_hom.length; j++) {
                pixel_hom[j][0] /= pixel_hom[j][2];
                pixel_hom[j][1] /= pixel_hom[j][2];
                pixel_hom[j][0] /= self.width;
                pixel_hom[j][1] /= self.height;
                pixel_hom[j].pop();
            }
            return pixel_hom;
        };

        self.projectPoint = function (px, py, pz, frame_id) {
            if (typeof frame_id === 'undefined') {
                frame_id = self.frame_id;
            } else if (!self.checkFrameId(frame_id)) {
                console.log("cameraModel -- Unknown Frame Id: ", frame_id);
                return;
            }
            if (frame_id === self.tfClient.fixedFrame.substring(1)) {
                var q = new THREE.Quaternion(self.transform.rotation.x,
                    self.transform.rotation.y,
                    self.transform.rotation.z,
                    self.transform.rotation.w);
                var tfMat = new THREE.Matrix4().makeRotationFromQuaternion(q);
                tfMat.setPosition(new THREE.Vector3(self.transform.translation.x,
                    self.transform.translation.y,
                    self.transform.translation.z));
                var pose = new THREE.Vector3(px, py, pz);
                tfMat.getInverse(tfMat);
                pose.applyMatrix4(tfMat);
                px = pose.x;
                py = pose.y;
                pz = pose.z;
            }
            var pixel_hom = numeric.dot(self.P, [[px],[py],[pz],[1]]);
            var pix_x = pixel_hom[0]/pixel_hom[2];
            var pix_y = pixel_hom[1]/pixel_hom[2];
            return [pix_x/self.width, pix_y/self.height];
        };
    };
    return module;
}) (RFH || {});

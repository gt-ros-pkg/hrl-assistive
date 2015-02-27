
//Old camera parameters
//  'Right Arm': {topic: '/r_forearm_cam/image_color_rotated',
//                optgroup:'Default',
//                cameraInfo: '/r_forearm_cam/camera_info',
//                clickable:true,
//                rotated: true,
//                width:640,
//                height:480},
//  'Left Arm': {topic: '/l_forearm_cam/image_color_rotated',
//               optgroup:'Default',
//               cameraInfo: '/l_forearm_cam/camera_info',
//               clickable: false,
//               rotated: true,
//               width:640,
//               height:480},
//  'AR Tag': {topic:'/ar_servo/confirmation_rotated',
//             optgroup:'Special',
//             cameraInfo:'/r_forearm_cam/camera_info',
//             clickable:false,
//             rotated: true,
//             width:640,
//             height:480},
//  'Head Registration': {topic: '/head_registration/confirmation',
//                        optgroup:'Special',
//                        cameraInfo: 'head_mount_kinect/rgb/camera_info',
//                        clickable: true,
//                        rotated: false,
//                        width: 640, //1280,
//                        height:480}//1024}

RFH.MjpegClient = function (options) {
    "use strict";
    var self = this;
    var options = options || {};
    self.imageTopic = options.imageTopic;
    self.containerId = options.containerId;
    self.divId = options.divId;
    self.imageId = options.imageId || self.divId + '-image';
    self.server = "http://"+options.host+":"+options.port;
    self.activeParams = {'quality': options.quality || 80,
                         'topic': options.imageTopic}

    self.cameraModel = new RFH.ROSCameraModel({ros: options.ros,
                                               infoTopic: options.infoTopic,
                                               rotated: options.rotated || false,
                                               tfClient: options.tfClient});
    self.refreshSize = function (resizeEvent) {
        if (!self.cameraModel.has_data) {
            $('#'+self.divId).html("Waiting on camera information.");
            return false;
        }
        var camRatio = self.cameraModel.width/self.cameraModel.height;
        var contWidth = $('#'+self.containerId).width();
        var contHeight = $('#'+self.containerId).height();
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
        return true;
    };
    $(window).on('resize', self.refreshSize);

    // Update the server and display to reflect the current properties
    self.update = function () {
        var srcStr = self.server+ "/stream"
        for (var param in self.activeParams)
        {
            srcStr += "?" + param + '=' + self.activeParams[param]
        }
        $("#"+self.imageId).attr("src", srcStr);
        $("#"+self.divId).width(self.activeParams['width'])
                         .height(self.activeParams['height']);
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

RFH.ROSCameraModel = function (options) {
    "use strict";
    var self = this;
    var options = options || {};
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
            self.tfClient.subscribe(self.camera.frame_id, function (tf) { self.transform = tf });
            console.log("Got camera data, subscribing to TF Frame: "+self.camera.frame_id);
        } else {
            console.log("No camera data -> no TF Transform");
            setTimeout(self.tryTFSubscribe, 500);
            }
    }

    self.infoMsgCB = function (infoMsg) {
        console.log("Updating camera model from " + self.infoTopic)
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
            self.tfClient.subscribe(self.frame_id, function (tf) { self.transform = tf });
            console.log("Subscribing to TF Frame: "+self.frame_id);
        } else {
            console.log("Camera at " + self.infoTopic + " reported empty frame id, cannot get TF data");
        }
        self.cameraInfoSubscriber.unsubscribe(); // Close subscriber to save bandwidth
        }
    
    self.infoSubCBList = [self.infoMsgCB];
    self.infoSubCB = function (msg) {
        for (var cb in self.infoSubCBList) {
           self.infoSubCBList[cb](msg); 
       }
    }
    self.updateCameraInfo = function () {
        // Re-subscribe to get new parameters
        self.has_data = false;
        self.cameraInfoSubscriber.subscribe(self.infoSubCB);
    }

    // back-project a pixel some distance into the real world
    // Returns a geoemtry_msgs/PointStamped msg
    self.projectPixel = function (px, py, dist) { 
       if (!self.has_data) { console.error("Camera Model has not received data from "+self.infoTopic); };
       var d = dist !== undefined ? dist : 2; 
       var pixel_hom = [[px],[py],[1]]; //Pixel value in homogeneous coordinates
       var vec = numeric.dot(self.KR_inv, pixel_hom);
       vec = numeric.transpose(vec)[0];
       var mag = numeric.norm2(vec);
       return numeric.mul(d/mag, vec);
    }

    self.projectPoint = function (px, py, pz, frame_id) {
        frame_id = typeof frame_id !== 'undefined' ? frame_id : self.frame_id;
        var fixedFrame = self.tfClient.fixedFrame.substring(1);
        if (frame_id[0] === '/') {
          frame_id = frame_id.substring(1);
        }
        if (frame_id !== fixedFrame && frame_id !== self.frame_id) {
            throw "cameraModel.projectPoint - Unknown frame_id"
            return;
        }
        if (frame_id === fixedFrame) {
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
        return [pix_x, pix_y];
        //return [pix_x/self.width, pix_y/self.height];
    }
};

var initMjpegCanvas = function (divId) {
    "use strict";
    $('#'+divId).off('click'); //Disable click detection so clickable_element catches it
    RFH.mjpeg = new RFH.MjpegClient({ros: RFH.ros,
                                     //imageTopic: '/head_mount_kinect/rgb/image_color',
                                     //infoTopic: '/head_mount_kinect/rgb/camera_info',
                                     imageTopic: '/head_mount_kinect/rgb_lowres/image',
                                     infoTopic: '/head_mount_kinect/rgb_lowres/camera_info',
                                     //imageTopic: '/head_wfov_camera/image_rect_color',
                                     //infoTopic: '/head_wfov_camera/camera_info',
                                     containerId: 'video-main',
                                     divId: 'mjpeg',
                                     host: RFH.ROBOT,
                                     port: 8080,
                                     quality: 85,
                                     tfClient:RFH.tfClient});
    RFH.mjpeg.cameraModel.infoSubCBList.push(RFH.mjpeg.refreshSize);
    RFH.mjpeg.cameraModel.updateCameraInfo();
};

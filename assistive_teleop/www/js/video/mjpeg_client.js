var MjpegClient = function (options) {
    var mjpegClient = this;
    var options = options || {};
    mjpegClient.divId = options.divId;
    mjpegClient.host = options.host;
    mjpegClient.port = options.port;
    mjpegClient.width = options.width || 640;
    mjpegClient.height = options.height || 480;
    mjpegClient.quality = options.quality || 90;

    mjpegClient.cameraTopics = {'Default':{
                                  'Head': '/head_mount_kinect/rgb/image_color',
                                  'Right Arm': '/r_forearm_cam/image_color_rotated',
                                  'Left Arm': '/l_forearm_cam/image_color_rotated'
                                },
                                'Special':{
                                  'AR Tag': '/ar_servo/confirmation_rotated',
                                  'Head Registration': '/head_registration/confirmation'
                                }
    }

    mjpegClient.activeParams = {'topic':mjpegClient.cameraTopics['Default']['Head'],
                                'width':mjpegClient.width,
                                'height':mjpegClient.height,
                                'quality':mjpegClient.quality}

    mjpegClient.server = "http://"+mjpegClient.host+":"+mjpegClient.port;
    mjpegClient.imageId = mjpegClient.divId + "Image";
    $("#"+mjpegClient.divId).html("<img id="+mjpegClient.imageId+"></img>");

    mjpegClient.update = function () {
        var srcStr = mjpegClient.server+ "/stream"
        for (param in mjpegClient.activeParams)
        {
            srcStr += "?" + param + '=' + mjpegClient.activeParams[param]
        }
        $("#"+mjpegClient.imageId).attr("src", srcStr);
    };

    // Set parameter value
    var setParam = function (param, value) {
      mjpegClient.activeParams[param] = value;
      mjpegClient.update();
    };

    // Return parameter value
    var getParam = function (param) {
      return mjpegClient.activeParams[param];
    };

    // Convenience function for back compatability to set camera topic
    var setCamera = function (topic) {
      mjpegClient.setParam('topic', topic);
    };
};

var initMjpegCanvas = function (divId) {
    window.mjpeg = new MjpegClient({'divId': divId,
                                    "host": window.ROBOT,
                                    "port": 8080,
                                    "width": 640,
                                    "height": 480,
                                    "quality": 85});
    window.mjpeg.update();    
    $('#'+divId).off('click');//,'mousedown','mouseup');
    window.clickableCanvas = new ClickableElement(divId);
    window.poseSender = new PoseSender(window.ros);
    window.p23DClient = new pixel23DClient(window.ros);
    initClickableActions();
};

//    var image_topics = []
//    var image_topics_names = []
//    var get_image_topics = function () {
//        var topicsClient = new window.ros.Service({
//            name: '/rosapi/topics_for_type',
//            serviceType: 'rosapi/TopicsForType'})
//        var req = new window.ros.ServiceRequest({type:'sensor_msgs/Image'})
//        topicsClient.callService(req, function (resp) {
//                for (topic in resp.topics) {
//                    if (resp.topics[topic].indexOf('/image_color') !== -1) {
//                        image_topics.push(resp.topics[topic]);
//                    }
//                }
//            var findKinect = function (topics_list) {
//                for (topic in topics_list) {
//                    if (topics_list[topic].indexOf('/rgb/') !== -1) {
//                        return topic
//                    }
//                }
//            }
//            window.mjpeg = new MjpegCanvas({
//            host:ROBOT,
//            port:8080,
//            topic : image_topics,
//            label : image_topics,
//            canvasID : 'mjpeg_canvas',
//            defaultStream: findKinect(image_topics),
//            width: 640,
//            height: 480,
//            quality: 70
//            })
//            })
//    }
//    get_image_topics()
//}

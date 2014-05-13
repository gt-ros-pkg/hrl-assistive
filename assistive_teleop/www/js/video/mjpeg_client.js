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
                                  'Head': '/head_mount_kinect/rgb/image_color;clickable',
                                  'Right Arm': '/r_forearm_cam/image_color_rotated',
                                  'Left Arm': '/l_forearm_cam/image_color_rotated'
                                },
                                'Special':{
                                  'AR Tag': '/ar_servo/confirmation_rotated',
                                  'Head Registration': '/head_registration/confirmation;clickable'
                                }
    }

    mjpegClient.activeParams = {'topic':mjpegClient.cameraTopics['Default']['Head'].split(';')[0],
                                'width':mjpegClient.width,
                                'height':mjpegClient.height,
                                'quality':mjpegClient.quality}

    mjpegClient.server = "http://"+mjpegClient.host+":"+mjpegClient.port;
    mjpegClient.imageId = mjpegClient.divId + "Image";
    $("#"+mjpegClient.divId).append("<img id="+mjpegClient.imageId+"></img>");

    mjpegClient.update = function () {
        var srcStr = mjpegClient.server+ "/stream"
        for (param in mjpegClient.activeParams)
        {
            srcStr += "?" + param + '=' + mjpegClient.activeParams[param]
        }
        $("#"+mjpegClient.imageId).attr("src", srcStr);
    };

    // Set parameter value
    mjpegClient.setParam = function (param, value) {
      mjpegClient.activeParams[param] = value;
      mjpegClient.update();
    };

    // Return parameter value
    mjpegClient.getParam = function (param) {
      return mjpegClient.activeParams[param];
    };

    // Convenience function for back compatability to set camera topic
    mjpegClient.setCamera = function (topic) {
      mjpegClient.setParam('topic', topic);
    };

    mjpegClient.getCameraMenu = function () {
        var html = "<select>";
        for (group in mjpegClient.cameraTopics) {
          html += "<optgroup label='"+group+"'>";
          for (cameraName in mjpegClient.cameraTopics[group]) {
            html += "<option value='" + mjpegClient.cameraTopics[group][cameraName] + "'>"+ cameraName + "</option>";
          }
          html += "</optgroup>";
        }
        return html;
    };
};

var initMjpegCanvas = function (divId) {
    var divRef = '#' + divId;
    $(divRef).append("<table>"+
                       "<tr><td colspan='4'><div id='mjpegDiv'></div></td></tr>" +
                       "<tr id='underVideoBar'>" + 
                         "<td style='text-align:right'>On Image Click:</td>" +
                         "<td id='image_click_select'></td>" + 
                         "<td style='text-align:right'>Camera:</td>" +
                         "<td id='cameraSelect'></td>" + 
                       "</tr>" +
                     "</table>");


    window.mjpeg = new MjpegClient({'divId': 'mjpegDiv',
                                    "host": window.ROBOT,
                                    "port": 8080,
                                    "width": 640,
                                    "height": 480,
                                    "quality": 85});
    window.mjpeg.update();    

    $('#cameraSelect').html(window.mjpeg.getCameraMenu());
    $('#cameraSelect').on('change', function() {
        window.mjpeg.setCamera( $('#cameraSelect :selected').val() );
      }
    );
    $('#'+divId).off('click');
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

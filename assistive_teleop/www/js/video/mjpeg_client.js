var initMjpegCanvas = function () {
    var image_topics = []
    var image_topics_names = []
    var get_image_topics = function () {
        var topicsClient = new window.ros.Service({
            name: '/rosapi/topics_for_type',
            serviceType: 'rosapi/TopicsForType'})
        var req = new window.ros.ServiceRequest({type:'sensor_msgs/Image'})
        topicsClient.callService(req, function (resp) {
                for (topic in resp.topics) {
                    if (resp.topics[topic].indexOf('/image_color') !== -1) {
                        image_topics.push(resp.topics[topic]);
                    }
                }
            var findKinect = function (topics_list) {
                for (topic in topics_list) {
                    if (topics_list[topic].indexOf('/rgb/') !== -1) {
                        return topic
                    }
                }
            }
            window.mjpeg = new MjpegCanvas({
            host:ROBOT,
            port:8080,
            topic : image_topics,
            label : image_topics,
            canvasID : 'mjpeg_canvas',
//            defaultStream: findKinect(image_topics),
            defaultStream: '/openni_kinect_head/rgb/image_color',
            width: 640,
            height: 480,
            quality: 70
            })
            window.clickableCanvas = new ClickableElement('mjpeg_canvas');
            window.poseSender = new PoseSender(window.ros);
            window.p23DClient = new pixel23DClient(window.ros);
            initClickableActions();
            })
    }
    get_image_topics()
}

var RFH = (function (module) {
    module.GetClickedPose = function(options) {
     'use strict';
        var self = this;
        var ros = options.ros;
        self.name = options.name || 'getClickedPoseAction';
        self.showButton = false;
        var container = options.container;
        var camera = options.camera;
        var pixel23d = new RFH.Pixel23DClient({
            ros: ros,
            cameraInfoTopic: camera.infoTopic
        });

        var $edges = $('.map-look');
        var $image = $('#mjpeg-image');

        var logResult = function (poseMsg) {
            console.log("Choose Spot id'd location: ", poseMsg);
        };

        var resultCBList = [{fn:logResult, once:false}];

        self.registerPoseCB = function (cbFn, once) {
            if (once === undefined) {
                once = false;
            } 
            resultCBList.push({fn:cbFn, once:once});
        };

        self.getPoseCBs = function (cbFn) {
            return resultCBList;
        };

        var removeOnceCalled = function (cb) {
            return !cb.once;
        };

        var poseCB = function(poseMsg) {
            $image.removeClass('cursor-wait');
            for (var i=0; i < resultCBList.length; i +=1) {
                resultCBList[i].fn(poseMsg);
            }
            resultCBList = resultCBList.filter(removeOnceCalled);
        };

        var clickCB = function(event, ui) {
            var pt = RFH.positionInElement(event);
            var px = (pt[0]/event.target.clientWidth);
            var py = (pt[1]/event.target.clientHeight);
            try {
                pixel23d.callRelativeScale(px, py, poseCB);
                $image.addClass('cursor-wait');
            } catch(err) {
                log(err);
            }
        };

        self.start = function() {
            $edges.addClass('visible').show();
            $image.addClass('cursor-select');
            $image.on('click.id-location', clickCB);
        };

        self.stop = function() {
            $edges.removeClass('visible').hide();
            $image.removeClass('cursor-select');
            $image.off('click.id-location');
        };
    };
    return module;
})(RFH || {});

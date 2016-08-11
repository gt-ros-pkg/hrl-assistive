RFH.ParamLocation = function(options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    self.name = options.name || 'getClickedPoseTask';
    self.showButton = false;
//    self.buttonText = 'Choose_Spot';
//    self.buttonClass = 'id-location-button';
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

    self.poseCB = function(poseMsg) {
        self.$image.removeClass('cursor-wait');
        for (var i=0; i < resultCBList.length; i +=1) {
            resultCBList[i].fn(poseMsg);
        }
        resultCBList = resultCBList.filter(removeOnceCalled);
    };

    clickCB = function(event, ui) {
        var pt = RFH.positionInElement(event);
        var px = (pt[0]/event.target.clientWidth);
        var py = (pt[1]/event.target.clientHeight);
        try {
            self.pixel23d.callRelativeScale(px, py, self.poseCB);
            self.$image.addClass('cursor-wait');
        } catch(err) {
            log(err);
        }
    };

    self.start = function() {
        self.$edges.addClass('visible').show();
        self.$image.addClass('cursor-select');
        self.$image.on('click.id-location', clickCB);
    };

    self.stop = function() {
        self.$edges.removeClass('visible').hide();
        self.$image.removeClass('cursor-select');
        self.$image.off('click.id-location');
    };
};

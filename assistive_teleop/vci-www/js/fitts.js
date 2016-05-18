var FITTS = {
    setup: function () {
        var test = new FittsLawTest();
        $('#startButton').button().on('click', test.run);
    }
};

var FittsLawTest = function (options) {
    'use strict';
    options = options || {};
    var self = this;
    var minD = options.targetMin || 10;
    var maxD = options.targetMax || 100;
    var $targetArea = $('#testArea');
    var data = [];
    var throughput;
    var err;

    var clickCB = function (event) {
        var clickTime = new Date();
        var d = event.target.data.dist;
        var w = event.target.data.width;
        var start = event.target.data.startTime;
        var time = clickTime - start
        updateThroughout(d, w, start, time);
        $targetArea.empty();
        $targetArea.append(generateTarget(event))
    };

    var calculateModel = function (data) {

    };

    var updateThroughout = function (dist, width, time) {
        data.push({d: dist, w: width, t: time}); 
        parameters = calculateModel(data);
    };

    var generateTarget = function(event){
        var mouseX = event.pageX;
        var mouseY = event.pageY;
        var h = $targetArea.height();
        var w = $targetArea.width();
        var centerX = maxD/2 + (w-maxD)*Math.random();
        var centerY = maxD/2 + (h-maxD)*Math.random();
        var dia = minD + (maxD-minD)*Math.random();
        var targetCreationTime = new Date();
        var dx = centerX - mouseX;
        var dy = centerY - mouseY;
        var distanceFromMouse = Math.sqrt(dx*dx + dy*dy);
        var targetData = {startTime: targetCreationTime,
                          dist: distanceFromMouse,
                          width: dia
                          };
        var target = $('<div>', {class: '.target', data: targetData});
        target.on('click', clickCB);
        return target;
    };

    var verticalBars = function () {
        var widths = [5, 10, 20];
        var dists = [0,10,20];
        for (var w=0; w<widths.length; w += 1) {
            for (var d=0; d<dists.length; d += 1) {
                var css = {position: 'relative',
                           height: widths[w]+'%',
                           width: '80%',
                           left: '10%',
                           dispay: 'block'};
                if (Boolean(d%2)) { 
                    css['top'] = dists[d]+'%';
                } else {
                    css['bottom'] = -dists[d]+'%';
                }
                $targetArea.append($("<div/>", {"class":"target"}).css(css));
            }
        }



    };

    self.run = function () {
        $('#startButton').remove();
       verticalBars(); 
    };

};

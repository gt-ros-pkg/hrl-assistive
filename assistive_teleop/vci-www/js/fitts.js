var FITTS = {
    setup: function () {
        var test = new FittsLawTest();
//        $('#startButton').button().on('click', test.run);
        test.run();
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

    var sphericalTargets = function () {
        var h = $targetArea.height();
        var w =  $targetArea.width();
        var cx = w/2;
        var cy = h/2;
        var widths = [20,50,100];
        var lim = Math.min(h, w)/2 - 0.75*widths[widths.length-1];
        var ringDiameters = [0.33*lim, 0.67*lim, lim]; 
        var n = 25;
        var targetSets = [];
        for (var iw=0; iw < widths.length; iw +=1) {
            for (var id=0; id < ringDiameters.length; id +=1) {
                targetSets.push(targetRing(cx, cy, n, ringDiameters[id], widths[iw]));
            }
        }
        return targetSets;
    };

    var ringIndexOrders = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25];
    var targetRing = function (cx, cy, n, diameter, width) {
        var angle, tx, ty, div;
        var targets = [];
        var css = {display: 'block',
                   position: 'absolute',
                   height: width+'px',
                   width: width+'px',
                   top: cy,
                   left: cx,
                   }
        for (var i=0; i<=n; i+=1) {
           angle = (2*Math.PI / n) * i + Math.PI;
           tx = cx + diameter * Math.sin(angle);
           ty = cy + diameter * Math.cos(angle);
           css['top'] = ty - width/2 + 'px';
           css['left'] = tx - width/2 + 'px';
           div = $("<div/>", {"class":"target sphere"}).css(css).hide();
           targets[ringIndexOrders[i]] = div;
           $targetArea.append(div);
        };
        return targets;
    };

    var timeDifferenceMS = function (date1, date2) {
       var dH = date2.getHours() - date1.getHours();
       var dM = date2.getMinutes() - date1.getMinutes();
       var dS = date2.getSeconds() - date1.getSeconds();
       var dMS = date2.getMilliseconds() - date1.getMilliseconds();
       return dMS + 1000*dS + 60*1000*dM + 60*60*1000*dH + 24*60*60*1000;
    };

    var indexOfDifficulty = function (event, target) {
        // TODO: Really do this;
        return 3;
    };

    var recordStartTime = function (event) {
        self.liveTarget.fittsData['start'] = new Date();
    };

    var nextTarget = function (event) {
        var clickTime = new Date();
        var dT = timeDifferenceMS(self.liveTarget.data['start'], clickTime);
        data.push({id: self.liveTarget.data['ID'],
                        time: dT});

        // Clear target, set next
        self.liveTarget.hide(); 
        self.liveTarget = self.allTargets.pop(0);
        self.liveTarget.fittsData = {};
        self.liveTarget.fittsData['id'] = indexOfDifficulty(event, self.liveTarget);
        $targetArea.one('mousemove', recordStartTime);
        self.liveTarget.show();
    };
    
    self.allTargets = [];
    self.liveTarget = null;

    self.run = function () {
        $('#startButton').remove();
        var targetSets = sphericalTargets(); 
        self.allTargets = [];
        for (var i=0; i<targetSets.length; i +=1) {
            var set = targetSets[i];
            for (var j=0; j<set.length; j +=1) {
                self.allTargets.push(set[j]);
            }
        }
        self.liveTarget = self.allTargets.pop(0);
        self.liveTarget.show();
        $targetArea.on('click', nextTarget);
    };

};

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
    var $targetArea = $('#testArea');
    var targetSets = [];
    var targets = [];
    var dataSets = [];
    var data = [];
    var liveTarget = null;
    var endTime;
    var startTime;

    var sphericalTargets = function () {
        var h = $targetArea.height();
        var w =  $targetArea.width();
        var cx = w/2;
        var cy = h/2;
        //var widths = [20,50,100];
        var widths = [100];
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

    var indexOfDifficulty = function (data) {
        var W = data['width'];
        var Dx = data['endXY'][0] - data['startXY'][0];
        var Dy = data['endXY'][1] - data['startXY'][1];
        var D = Math.sqrt(Dx*Dx + Dy*Dy);
        var Ex = data['endXY'][0] - data['goalXY'][0];
        var Ey = data['endXY'][1] - data['goalXY'][1];
        var E = Math.sqrt(Ex*Ex + Ey*Ey);
        var ID = Math.log2((D/W) + 1);



    };

    var responseCB = function (event) {
        var clickTime = new Date();
        if (liveTarget.fittsData['startTime'] === undefined) { return; } // Probably haven't moved, just ignore the click...
        liveTarget.fittsData['time'] = timeDifferenceMS(liveTarget.fittsData['startTime'], clickTime);
        liveTarget.fittsData['endXY'] = [event.pageX, event.PageY];
        data.push(liveTarget.fittsData);
        liveTarget.remove(); 
        nextTarget();
    };
    
    var recordStart = function (event) {
        liveTarget.fittsData['startTime'] = new Date();
        liveTarget.fittsData['startXY'] = [event.pageX, event.pageY];
    };

    var nextTarget = function () {
        if (targets.length <= 0) {
            newTargetSet();    
        }
        liveTarget = targets.pop(0);
        liveTarget.fittsData = {};
        $targetArea.one('mousemove', recordStart);
        liveTarget.show();
        var pos = liveTarget.position();
        var w = liveTarget.width();
        var h = liveTarget.height();
        liveTarget.fittsData['width'] = w; // Assumes circular target
        liveTarget.fittsData['goalXY'] = [pos['left']+w/2, pos['top']+h/2];
    }

    var newTargetSet = function () {
        if (data.length > 0) { // Add data from last set and clear for all but first call
            dataSets.push(data);
            data = [];
        }
        if (targetSets.length <= 0) {
            endTime = new Date();
            finishAnalysis();
        } else {
            targets = targetSets.pop();
        }
    };


    self.run = function () {
        targetSets = sphericalTargets(); 
        $('#startButton').remove();
        nextTarget();
        startTime = new Date();
        $targetArea.on('mousedown', responseCB);
    };

    var removeOutliers = function (data) {
        var distances = getDistances(data);
        var times = getTimes(data);
        var timeMean = math.mean(times);
        var timeStd = math.std(times);
        var minTime = timeMean - 3*timeStd; 
        var maxTime = timeMean + 3*timeStd;
        var distMean = math.mean(distances);
        var distStd = math.std(distances)
        var minDist = distMean - 3*distStd;
        var maxDist = distMean + 3*distStd;
        var timeMinOutliers = math.smaller(times, minTime);
        var timeMaxOutliers = math.larger(times, maxTime);
        var distMinOutliers = math.smaller(distances, minDist);
        var distMaxOutliers = math.larger(distances, maxDist);
        var outliers = math.or(timeMinOutliers, timeMaxOutliers)
        outliers = math.or(outliers, distMinOutliers);
        outliers = math.or(outliers, distMaxOutliers);
        var filteredData = [];
        for (var j=0; j<data.length; j += 1) {
            if (!outliers[i]) {
                filteredData.push(data[j]);
            } else {
                console.log("Removed outlier");
            };
        };
        return filteredData;
    }

    var getTimes = function (data) {
        var times = [];
        for (var i=0; i < data.length; i += 1) {
            times.push(data[i]['time']);
        }
        return times;
    };

    var getDistances = function (data) {
        var distances = [];
        for (var i=0; i < data.length; i += 1) {
            distances.push(math.norm([data[i]['endXY'][0] - data[i]['startXY'][0],
                                      data[i]['endXY'][1] - data[i]['startXY'][1] ]));
        };
        return distances;
    };

    var finishAnalysis = function () {
        //TODO: Process datasets for results
        $targetArea.off('mousedown').css({'background-color':'orange'});
        var data;
        for (var i=0; i<dataSets.length; i += 1) {
            data = removeOutliers(dataSets[i]);

            directionalErrors = [];
            for (var j=0; j<data.length; j +=1) {
               var Vse = [ data['endXY'][0] - data['startXY'][0], data['endXY'][1] - data['startXY'][1] ];
               var Vsg = [ data['goalXY'][0] - data['startXY'][0], data['goalXY'][1] - data['startXY'][1] ];
               var err = ( ( math.dot(Vse, Vsg) / math.dot(Vsg, Vsg) ) * math.norm(Vge) ) - math.norm(Vge);
               directionalErrors.push(err);
            }
            var We = 4.133*math.std(directionalErrors);
            var dists = getDistances(data);
            var De = math.mean(dists);
            var IDe = math.log((De/Ww)+1 , 2);


            // Compute effective width separately for each dataset

        }
        // combine list of (time, IDe) values for all datasets
        // Least-squares fit, calculate throughput.
    };

    var displayResults = function (results) {
        var html = "";
        $targetArea.css({'background-color':'LightGreen'}).innerHTML(html);
    };

};

var FittsLawTest = function (options) {
    'use strict';
    options = options || {};
    var self = this;
    var dwellTime = options.dwellTime || 0;
    var $targetArea = $('#testArea');
    var $roundCount = $('#roundCount');
    var setParameters = [];
    var targetSets = [];
    var targets = [];
    var dataSets = [];
    var data = [];
    var liveTarget = null;
    var targetDistance = null;
    var endTime;
    var startTime;
    var round = 0;
    var nRounds = null;

    var queryString = window.location.search;
    queryString = queryString.replace('?','');
    var array = queryString.split('&');
    var dict = {};

    for (var i=0; i<array.length; i += 1){
        var part = array[i];
        var key = part.split('=')[0];
        var value = part.split('=')[1];
        dict[key] = value;
    }
    var userID = dict.uid;
    var SUBSET = dict.set - 1;

    var shuffleArray = function (array) {
        var newArray = [];
        while (array.length > 0) {
            // Take first element
            if (array.length>0){
                newArray.push(array.splice(0,1)[0]);
            }
            // Take last element
            if (array.length>0){
                newArray.push(array.splice(array.length-1,1)[0]);
            }
            // Take middle element
            if (array.length>0){
                newArray.push(array.splice(Math.floor(array.length/2),1)[0]);
            }
        }
        return newArray;
    };

    var sphericalTargets = function () {
        var h = $targetArea.height();
        var w =  $targetArea.width();
        var cx = w/2;
        var cy = h/2;

        var bound = Math.min(h,w)/2;
        var lim = 0.75*bound;

        var widths = [0.05*bound, 0.125*bound, 0.25*bound];
        var ringDiameters = [0.25*lim, 0.5*lim, 0.75*lim, lim]; 
        var n = 25;
        var targetSets = [];
        for (var iw=0; iw < widths.length; iw +=1) {
            for (var id=0; id < ringDiameters.length; id +=1) {
                setParameters.push([ringDiameters[id], widths[iw]]);
            }
        }
        setParameters = shuffleArray(setParameters); // Randomize the order of cases
        for (var i=0; i<setParameters.length; i += 1) {
            targetSets.push(targetRing(cx, cy, n, setParameters[i][0], setParameters[i][1]));
        }
        if (SUBSET !== null) {
            setParameters = setParameters.slice(3*SUBSET, 3*SUBSET+3);
            targetSets = targetSets.slice(3*SUBSET, 3*SUBSET + 3);
        }
        nRounds = targetSets.length;
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
        };
        for (var i=0; i<=n; i+=1) {
            angle = (2*Math.PI / n) * i + Math.PI;
            tx = cx + diameter * Math.sin(angle);
            ty = cy + diameter * Math.cos(angle);
            css.top = ty - width/2 + 'px';
            css.left = tx - width/2 + 'px';
            div = $("<div/>", {"class":"target sphere"}).css(css).hide();
            targets[ringIndexOrders[i]] = div;
            $targetArea.append(div);
        }
        return targets;
    };

    var timeDifferenceMS = function (date1, date2) {
        var dH = date2.getHours() - date1.getHours();
        var dM = date2.getMinutes() - date1.getMinutes();
        var dS = date2.getSeconds() - date1.getSeconds();
        var dMS = date2.getMilliseconds() - date1.getMilliseconds();
        var timeDiff = dMS + 1000*dS + 60*1000*dM + 60*60*1000*dH;
        var withoutDwell = timeDiff - dwellTime;
        return withoutDwell;
    };

    var responseCB = function (event) {
        var clickTime = new Date();
        if (liveTarget.fittsData.startTime === undefined) { return; } // Probably haven't moved, just ignore the click...
        liveTarget.fittsData.endTime = clickTime;
        liveTarget.fittsData.duration = timeDifferenceMS(liveTarget.fittsData.startTime, clickTime);
        liveTarget.fittsData.endXY = [event.pageX, event.pageY];
        data.push(liveTarget.fittsData);
        liveTarget.remove(); 
        nextTarget();
    };

    var recordStart = function (event) {
        liveTarget.fittsData.startTime = new Date();
        liveTarget.fittsData.startXY = [event.pageX, event.pageY];
    };

    var nextTarget = function () {
        if (targets.length <= 0) {
            newTargetSet();    
            return;
        }
        liveTarget = targets.pop(0);
        liveTarget.fittsData = {};
        $targetArea.one('mousemove', recordStart);
        liveTarget.show();
        var pos = liveTarget.position();
        var w = liveTarget.width();
        var h = liveTarget.height();
        liveTarget.fittsData.width = w; // Assumes circular target
        liveTarget.fittsData.goalXY = [pos.left+w/2, pos.top+h/2];
    };

    var newTargetSet = function () {
        if (data.length > 0) { // Add data from last set and clear for all but first call
            dataSets.push(data);
            data = [];
        }
        if (targetSets.length <= 0) {
            endTime = new Date();
            $targetArea.off('mousedown');
            finishAnalysis();
        } else {
            $targetArea.off('mousedown');
            round += 1;
            $roundCount.html('Completed round '+round+' of '+nRounds+'.</br></br>You may pause between rounds (now)').show();
            $nextRoundButton.show();
            targets = targetSets.pop();
        }
    };

    var startRound = function () {
        $nextRoundButton.hide();
        $roundCount.hide();
        $targetArea.on('mousedown', responseCB);
        nextTarget();
    };

    var $nextRoundButton = $('<button class="startButton">Start Next Round</button>').button().on('click', startRound).hide().css({bottom:'initial', top:'50%'}).on('dragstart', function () {return false;});

    self.run = function () {
        targetSets = sphericalTargets(); 
        targets = targetSets.pop();
        $targetArea.append($nextRoundButton);
        startTime = new Date();
        startRound();
    };

    var removeOutliers = function (data) {
        var distances = getDistances(data);
        var times = getTimes(data);
        var timeMean = math.mean(times);
        var timeStd = math.std(times);
        var minTime = timeMean - 3*timeStd; 
        var maxTime = timeMean + 3*timeStd;
        var distMean = math.mean(distances);
        var distStd = math.std(distances);
        var minDist = distMean - 3*distStd;
        var maxDist = distMean + 3*distStd;
        var timeMinOutliers = math.smaller(times, minTime);
        var timeMaxOutliers = math.larger(times, maxTime);
        var distMinOutliers = math.smaller(distances, minDist);
        var distMaxOutliers = math.larger(distances, maxDist);
        var outliers = math.or(timeMinOutliers, timeMaxOutliers);
        outliers = math.or(outliers, distMinOutliers);
        outliers = math.or(outliers, distMaxOutliers);
        var filteredData = [];
        for (var i=0; i<data.length; i += 1) {
            if (!outliers[i]) {
                filteredData.push(data[i]);
            } else {
                console.log("Removed outlier");
            }
        }
        return filteredData;
    };

    var getTimes = function (data) {
        var times = [];
        for (var i=0; i < data.length; i += 1) {
            times.push(data[i].duration);
        }
        return times;
    };

    var distance = function (pt1, pt2) {
        return math.norm([pt2[0] - pt1[0], pt2[1] - pt1[1]]);
    };

    var getDistances = function (data) {
        var distances = [];
        for (var i=0; i < data.length; i += 1) {
            distances.push(distance(data[i].endXY, data[i].startXY));
        }
        return distances;
    };

    var finishAnalysis = function () {
        var MTs = [];
        var IDes = [];

        var data;
        for (var i=0; i<dataSets.length; i += 1) {
            data = removeOutliers(dataSets[i]);

            var directionalErrors = [];
            for (var j=0; j<data.length; j +=1) {
                var Vse = [ data[j].endXY[0] - data[j].startXY[0], data[j].endXY[1] - data[j].startXY[1] ];
                var Vsg = [ data[j].goalXY[0] - data[j].startXY[0], data[j].goalXY[1] - data[j].startXY[1] ];
                var err = math.norm(Vsg) - ( ( math.dot(Vse, Vsg) / math.dot(Vsg, Vsg) ) * math.norm(Vsg) );
                directionalErrors.push(err);
            }
            var err_std = math.std(directionalErrors);
            var We = 4.133*err_std;
            var dists = getDistances(data);
            var De = math.mean(dists);
            var IDe = math.log((De/We)+1 , 2);
            IDes.push(IDe);
            var times = getTimes(data);
            var MT = math.mean(times);
            MTs.push(MT);
        }
        console.log("MTs: ", MTs);
        console.log("IDes: ", IDes);
        var TP = math.round(throughput(MTs, IDes), 3);
        var fittsCoefficients = leastSquares(IDes, MTs);
        var a = math.round(fittsCoefficients.intercept, 4);
        var b = math.round(fittsCoefficients.slope, 4);
        displayResults(a, b, TP,  dataSets);
    };

    var displayResults = function (a, b, TP, dataSets) {
        var time = new Date();
        var dataFileName = "FittsLawResults-"+time.getDate()+'-'+(time.getMonth()+1).toString()+'-'+time.getUTCFullYear()+'-'+time.getHours()+'-'+time.getMinutes()+'-'+time.getSeconds() + '.txt';
        //added for saving data
        dataSendFunc(a, b, TP, dataSets, dataFileName);
        //var resultURL = makeResultsFile(a, b, TP, dataSets, dataFileName);
        //var $saveButton = $('<div class="startButton">Save data</div>').button();
        //$saveButton.on('click', function(event){dataSendFunc(a, b, TP, dataSets, dataFileName);});
        //$targetArea.append($saveButton);

        // Display thank you msg
        //html = "<h2>Test Complete.</h2>";
        //html += "<h3>This test is now complete, and your results have been saved.</h3>";
        //html += "<h3>Thank you for your participation.</h3>";
        //html += "<h3>You may now close this window, or click 'Start Over' to take the test again if you wish.</h3>";
//        html += "<h3>Thank you!</h3>";
//        html += "<h3>Fitts Law Model: <span style='color:red'>MT = " + a.toString() + " + " + b.toString() + " x IDe</span></h3>";
//        html += "<h3>Throughput: <span style='color:red'>"+TP+"</span></h3>";
//        html += "<h3>Please click below to download your results, and e-mail them back to me!</h3>";
//        html += "<p>For more information, check out the <a href='https://en.wikipedia.org/wiki/Fitts%27s_law'>Wikipedia entry on Fitt's Law</a></p>";
        $targetArea.css({'background-color':'LightGreen'});
        $targetArea.append("<br>");		
        var $refreshButton = $('<a class="startButton">Start over</a>').button().on('click', function(){location.reload(true);}).hide();
        $targetArea.append($refreshButton);


    };

    var leastSquares = function (X,Y) {
        var sum_x = math.sum(X);
        var sum_y = math.sum(Y);
        var sum_xx = math.dot(X,X);
        var sum_xy = math.dot(X,Y);
        var N = X.length;
        var slope = (N*sum_xy - sum_x*sum_y) / (N*sum_xx - sum_x*sum_x);
        var intercept = (sum_y/N) - slope * (sum_x/N);
        return {slope:slope, intercept:intercept};
    };

    var throughput = function (MTlist, IDlist) {
        var tp = 0;
        for (var i=0; i<MTlist.length; i += 1) {
            tp += (IDlist[i] / MTlist[i]);
        }
        return 1000 * (tp / MTlist.length); // time in ms. 1000 gives bits/second.
    };

    var resultsFile = null;
    var makeResultsFile = function (a, b, TP, dataSets, dataFileName) {
        data = {'filename': dataFileName,
            'a': a,
            'b': b,
            'throughput': TP,
            'dwellTime': dwellTime,
            'startTime': startTime,
            'endTime': endTime,
            'setParameters': setParameters,
            'dataSets': dataSets};
            var contents = new Blob([JSON.stringify(data)], {type:'text/plain'});
            if (resultsFile !== null) {
                window.URL.revokeObjectURL(resultsFile);
            }
            resultsFile = window.URL.createObjectURL(contents);
            return resultsFile;
    };

    var dataSendFunc = function (a, b, TP, dataSets, dataFileName) {
        var data = {'filename': dataFileName,
            'userID': userID,
            'a': a,
            'b': b,
            'throughput': TP,
            'dwellTime': dwellTime,
            'startTime': startTime,
            'endTime': endTime,
            'setParameters': setParameters,
            'dataSets': dataSets};

            $.ajax({method: "POST",
                   url: "cgi-bin/save.py",
                   data:  data
                    }
              ).done(function (response) {
                      $targetArea.html(response).css({'background-color':'green'});
                      $('.startButton').button();
                     }
              );
    };
};

var FITTS = {
    setup: function () {
        var startFittsLaw = function () {
            var dwellDelay = 1000*$('#dwellTimeInput').val();
            var test = new FittsLawTest({dwellTime: dwellDelay});
            $('#testArea').css({'background-color':'Gray'}).empty();
            test.run();
        };
        $('button.startButton').button().on('click', startFittsLaw);
    }
};

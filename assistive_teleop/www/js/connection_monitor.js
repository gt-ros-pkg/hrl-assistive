RFH.ConnectionMonitor = function (options) {
    'use strict';
    var self = this;
    self.divId = options.divId || "network_status";
    self.target = options.target || "./bwTest50k.txt";
    self.targetSize = options.targetSize || 50000;
    self.bandwidthMeasures = [];
    self.baseTarget = options.baseTarget || "./bwTest10b.txt";
    self.baseTimes = [1];
    self.baseAvg = 0.00001;
    self.timer = null;

    self.successCB = function (data, textStatus, jqXHR) {
        var tripTime = new Date() - self.sendTime;
        tripTime -= self.baseAvg;
        var bw = 0.000001 * (1/((tripTime/1000) / self.targetSize)); //MB per s
        self.bandwidthMeasures.push(bw);
        if (self.bandwidthMeasures.length > 10) {
            self.bandwidthMeasures.splice(0,1);
        }
        var sum = 0;
        for (var i=0; i < self.bandwidthMeasures.length; i += 1) {
            sum += self.bandwidthMeasures[i];
        };
        var avg = sum/self.bandwidthMeasures.length;
        $('#'+self.divId).html(avg.toFixed(1) + 'MB/s');
    };

    self.baseSuccessCB = function (data, textStatus, jqXHR) {
        var tripTime = new Date() - self.baseSendTime;
        self.baseTimes.push(tripTime);
        if (self.baseTimes.length > 10) {
            self.baseTimes.splice(0,1);
        }
        var sum = 0;
        for (var i=0; i < self.baseTimes.length; i += 1) {
            sum += self.baseTimes[i];
        };
        self.baseAvg = sum/self.baseTimes.length;

    };

    self.errorCB = function (jqXHR, textStatus, errorThrown) {
        console.log("Error in Ajax request");
    };

    self.checkBandwidth = function () {
        self.baseSendTime = new Date();
        $.ajax({url: self.baseTarget,
                success: self.baseSuccessCB,
                error: self.errorCB,
                dataType: 'text',
                cache: false});

        self.sendTime = new Date();
        $.ajax({url: self.target,
                success: self.successCB,
                error: self.errorCB,
                dataType: 'text',
                cache: false});
    };

    self.start = function () {
        self.timer = setInterval(self.checkBandwidth, 2000);
    };
};

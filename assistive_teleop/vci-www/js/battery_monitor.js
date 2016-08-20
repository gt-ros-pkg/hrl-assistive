RFH.BatteryMonitor = function (options) {
    'use strict';
    var self = this;
    var ros = options.ros;
    self.div = options.div || 'battery-status';
    self.divId = '#'+self.div;

    self.width = $(self.divId).width()
    self.contactWidth = $(self.divId + ' .battery-contact').width();
    self.mainWidth = self.width - self.contactWidth;
    self.contactPct = 100 * (self.contactWidth/self.width);
    self.mainPct = 100 * (self.mainWidth/self.width);

    var times_left = [null, null, null, null];
    var charge_pcts = [null, null, null, null];
    var power_present_array = [null, null, null, null];

    var current_charge = null;
    var ac_present = null;
    var duration = null;

    var updateDisplay = function () {
        var grad_pct, txt;
        var disp = ac_present ? 'block' : 'none';
        $(self.divId + ' .charge-bolt').css('display', disp);
        $(self.divId + ' .charge-bolt').css('display', disp);

        if (current_charge > self.mainPct) {
            grad_pct = 100 * ((current_charge - self.mainPct) / (self.contactPct));
            $(self.divId + ' .battery-main').css({'background':'', 'background-color':"rgba(30,200,45,0.87)"});
            $(self.divId + ' .battery-contact').css('background', "linear-gradient(to left, rgba(30,200,45,0.87)"+
                                                                   grad_pct + "%, LightGray " + grad_pct + "%)");
        } else {
            grad_pct = self.mainWidth * (current_charge/self.mainPct);
            $(self.divId + ' .battery-contact').css({'background':'', 'background-color':'LightGray'});
            $(self.divId + ' .battery-main').css('background', "linear-gradient(to left, rgba(30,200,45,0.87)"+
                                                               grad_pct + "%, LightGray " + grad_pct + "%)");
        }
        if (ac_present && duration > 24*3600) {
            txt = "Full"
        } else {
            var hrs = Math.floor(duration/3600);
            var mins = Math.floor((duration - 3600*hrs)/60);
            if (mins < 10) { mins = '0'+mins.toString() };
            txt = hrs + ":" + mins;
        }
        $(self.divId + ' span').html(txt);
    };

    var notNullFilter = function (item) {
        return item !== null;
    };

    var updateNetState = function () {
        var i, tmp;
        // Get average of known charge values
        var known_charges = charge_pcts.filter(notNullFilter);
        tmp = 0;
        for (i=0; i<known_charges.length; i+=1) {
           tmp += known_charges[i];
        }
        current_charge = tmp / known_charges.length;

        // Get average of known times
        var known_times = times_left.filter(notNullFilter);
        tmp = 0;
        for (i=0; i<known_times.length; i+=1) {
           tmp += known_times[i];
        }
        duration = tmp / known_times.length;

        // Get power present guess
        var known_power = power_present_array.filter(notNullFilter);
        tmp = 0;
        for (i=0; i<known_power.length; i+=1) {
           tmp += known_power[i];
        }
        ac_present = (tmp / known_power.length) >= 0.5

        updateDisplay();
    };

    var getMsgData = function (msg) {
        times_left[msg.id] = msg.time_left.secs;
        charge_pcts[msg.id] = msg.average_charge;
        for (var i=0; i<msg.battery.length; i+=1) {
            if (msg.battery[i].power_present && !msg.battery[i].power_no_good) {
                power_present_array[msg.id] = true;
                break;
            }
            power_present_array[msg.id] = false;
        };
        updateNetState();
    };

    self.batteryCBList = [getMsgData];
    var batterySubCB = function (msg) {
        for (var i = 0; i < self.batteryCBList.length; i += 1) {
          self.batteryCBList[i](msg);
        }
    };

    var batterySub = new ROSLIB.Topic({
        ros: ros,
        name: 'battery/server2',
        messageType: 'pr2_msgs/BatteryServer2'
    })
    batterySub.subscribe(batterySubCB);
}

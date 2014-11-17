RFH.BatteryMonitor = function (options) {
    'use strict';
    var self = this;
    self.ros = options.ros;
    self.div = options.div || 'battery-status';
    self.divId = '#'+self.div;


    self.width = $(self.divId).width()
    self.contactWidth = $(self.divId + ' .battery-contact').width();
    self.mainWidth = self.width - self.contactWidth;
    self.contactPct = 100 * (self.contactWidth/self.width);
    self.mainPct = 100 * (self.mainWidth/self.width);

    self.current_charge = 0;
    self.ac_present = false;
    self.duration = 0;


    self.updateDisplay = function (chargePct, AC, duration) {
        var grad_pct;
        var disp = AC ? 'block' : 'none';
        $(self.divId + ' .charge-bolt').css('display', disp);

        if (chargePct > self.mainPct) {
            grad_pct = 100 * ((chargePct - self.mainPct) / (self.contactPct));
            $(self.divId + ' .battery-main').css({'background':'', 'background-color':"rgba(30,200,45,0.87)"});
            $(self.divId + ' .battery-contact').css('background', "linear-gradient(to left, rgba(30,200,45,0.87)"+
                                                                   grad_pct + "%, LightGray " + grad_pct + "%)");
        } else {
            grad_pct = self.mainWidth * (chargePct/self.mainPct);
            $(self.divId + ' .battery-contact').css({'background':'', 'background-color':'LightGray'});
            $(self.divId + ' .battery-main').css('background', "linear-gradient(to left, rgba(30,200,45,0.87)"+
                                                               grad_pct + "%, LightGray " + grad_pct + "%)");
        }
        var hrs = Math.floor(duration/3600);
        var mins = Math.ceil((duration - 3600*hrs)/60);
        $(self.divId + ' span').html(hrs+":"+mins);
    }

    self.batterySubCB = function (msg) {
        self.current_charge = msg.relative_capacity;
        self.ac_present = msg.AC_present > 0 ? true : false;
        self.duration = msg.time_remaining.secs;
        self.updateDisplay(self.current_charge, self.ac_present, self.duration);
    }
    self.batteryCBList = [self.batterySubCB];
    self.batterySub = new ROSLIB.Topic({
        ros: self.ros,
        name: 'power_state',
        messageType: 'pr2_msgs/PowerState'
    })
    self.batterySub.subscribe(function (msg) {
        for (var i = 0; i < self.batteryCBList.length; i += 1) {
          self.batteryCBList[i](msg);
        }
    });
}

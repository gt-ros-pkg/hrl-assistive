var RunStop = function (ros) {
    var runStop = this;
    runStop.ros = ros;
    runStop.stopService = new runStop.ros.Service({
        name:'/pr2_etherCAT/halt_motors',
        serviceType: 'std_srvs/Empty'});
    runStop.stop = function () {
            runStop.stopService.callService({}, function () {});
        };
    runStop.runService = new runStop.ros.Service({
        name:'/pr2_etherCAT/reset_motors',
        serviceType: 'std_srvs/Empty'});
    runStop.start = function () {
            runStop.runService.callService({}, function () {});
        };
};

var initRunStop = function (divId) {
    var runStop = new RunStop(window.ros);
    $('#'+divId).append('<button id="'+divId+'StopButton">'+
                        '<button id="'+divId+'RunButton">');
    $('#'+divId+'StopButton').html('STOP ROBOT');
    $('#'+divId+'StopButton').css({'color':'red',
                                   'background-color':'gray',
                                   'font':'bold',
                                   'border-style':'none',
                                   'height':'30px',
                                   'width':'100%'});
    $('#'+divId+'StopButton').click(function () {runStop.stop();
                                                $(':button:visible').fadeTo(0,0.4);
                                                $('#'+divId+'RunButton').show();
                                                log("YOU'VE STOPPED THE ROBOT!"+
                                                "Please click 'REACTIVATE' to reset.");
    });
    $('#'+divId+'RunButton').html('Resume').hide();
    $('#'+divId+'RunButton').css({'color':'white',
                                   'background-color':'green',
                                   'font':'bold',
                                   'height':'30px',
                                   'width':'100%'});
    $('#'+divId+'RunButton').click(function () {runStop.start();
                                    $(':button:visible').fadeTo(0,1);
                                    $(this).hide();
                                    log('Operation Resumed');
    });
};

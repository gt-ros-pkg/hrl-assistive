var assistive_teleop = {
  start :  function () {
    window.ROBOT = window.location.host.split(':')[0];//Use localhost when serving website directly from robot 
    window.PORT = '9091';//Must match port on which rosbridge is being served
    window.log = function (message) {
        $('#console').html("<big><strong>" + message.toString() + "</strong></big>"); 
        console.log("Log to user: " + message.toString());
    };

    $('#tabs').css({'top':'0px'})
    var tabs = $("#tabs").tabs();
    tabs.find(".ui-tabs-nav").sortable({axis:"xy", stop: function() {tabs.tabs("refresh");}});
    $('#cont_sel_container').buttonset();
    $('label:first','#cont_sel_container').removeClass('ui-corner-left')
                                          .addClass('ui-corner-top centered');
    $('label:last','#cont_sel_container').removeClass('ui-corner-right')
                                         .addClass('ui-corner-bottom centered');
    $('#scale_slider').slider({
        value:0.5,
        min:0,
        max:1.0,
        step:0.01,
        orientation:'vertical'}); 
    $('.bpd, #cart_controller, .ar_servo_button, .traj_play_cont,'+
      '#adj_mirror, #traj_play_reverse, #ell_controller, #reg_head,'+
      '#rezero_wrench, #send_shave_select, #shave, #shave_stop, #tool_power').button();

    window.ros = new ROS('ws://'+ ROBOT + ':'+ PORT);
    window.ros.on('close', function(e) {
        log("Disconnected or Can't Connect to " + ROBOT + ":"+ PORT + ".");
      }
    );
    ros.on('error', function(e) {
      log("Rosbridge Connection Error!");
      }
    );
    ros.on('connection', function(e) {
        log("Connected to " + ROBOT + ".");
        extendROSJS();
        initMjpegCanvas();
        initPr2(); 
        initGripper('horizontal');
        initTorsoSlider('horizontal');
        initTTS('tabTTS');
        initARServoTab('tabServoing');
        initRunStop('runStopDiv');
        initTrajPlay();
        initBodyRegistration('tabBodyReg');
        initEllControl();
        initCartControl();
        initTaskInterface('tabTasks');
        if (window.location.hash.search('ft') !== -1) {
          initFTDisplay('FTDisplay', {});
        }
        if (window.location.hash.search('skin') !== -1) {
          initSkinUtils();
        }
        teleopHead();

      }
    );
  }
};

var assistive_teleop = {
  checkBrowser: function () {
      var ua = navigator.userAgent;
      var idx = ua.indexOf('Chrome/');
      if (idx === -1) {
          $("body").replaceWith("<body><p>Please use Google Chrome</p></body>")
          alert("Please Use Google Chrome");
          return false;
      };
      var vm = ua.slice(idx+7, idx+9);
      if (vm <= 30) {
          $("body").replaceWith("<body><p>Please update your Chrome Browser.</p></body>")
          alert("Please update your Chrome Browser.");
          return false;
      };
      return true;
  },

  start :  function () {
    var good_browser = this.checkBrowser();
    if (!good_browser) {
    return;
    }

    this.ROBOT = window.location.host.split(':')[0];//Use localhost when serving website directly from robot 
    this.PORT = '9091';//Must match port on which rosbridge is being served
    initUserLog('#console')

    $('#tabs').css({'top':'0px'})
    var tabs = $("#tabs").tabs();
    $('#cont_sel_container').buttonset();
    $('label:first','#cont_sel_container').removeClass('ui-corner-left')
                                          .addClass('ui-corner-top centered');
    $('label:last','#cont_sel_container').removeClass('ui-corner-right')
                                         .addClass('ui-corner-bottom centered');
    $('#scale_slider').slider({value:0.5, min:0, max:1.0,
                               step:0.01, orientation:'vertical'}); 
    $('.bpd, #cart_controller, .ar_servo_button, .man_task_cont, .traj_play_cont,'+
      '#adj_mirror, #traj_play_reverse, #ell_controller, #reg_head,'+
      '#rezero_wrench, #send_shave_select, #shave, #shave_stop, #tool_power').button();

    this.ros = new ROS('ws://'+ this.ROBOT + ':'+ this.PORT);
    this.ros.on('close', function(e) {
        log("Disconnected or Can't Connect to " + this.ROBOT + ":"+ this.PORT + ".");
      }
    );
    this.ros.on('error', function(e) {
      log("Rosbridge Connection Error!");
      }
    );
    this.ros.on('connection', function(e) {
        log("Connected to " + assistive_teleop.ROBOT + ".");
        extendROSJS(assistive_teleop.ros);
        initMjpegCanvas('videoAndControls');
        initClickableActions();
        initPr2(); 
        initGripper('horizontal');
        initTorsoSlider('horizontal');
        //initTTS('tabTTS');
        //initARServoTab('tabServoing');
        initRunStop('runStopDiv');
        initTrajPlay();
        //initBodyRegistration('tabBodyReg');
        //initEllControl();
        initCartControl();
        initTaskInterface('tabTasks');
        //initRYDSTab('tabRYDS');
        initManTaskTab();
        initAdGUI('scooping');
        initAdGUI('feeding');
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

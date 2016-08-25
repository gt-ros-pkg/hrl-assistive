var RFH = {
    positionInElement: function (e) {
        var posx = 0;
        var posy = 0;
        if (e === undefined) {e = window.event;}
        if (e.pageX || e.pageY) 	{
            posx = e.pageX;
            posy = e.pageY;
        }
        else if (e.clientX || e.clientY) 	{
            posx = e.clientX + document.body.scrollLeft +
                 document.documentElement.scrollLeft;
            posy = e.clientY + document.body.scrollTop +
                 document.documentElement.scrollTop;
        }
        var offsetLeft = 0;
        var offsetTop = 0;
        var element = document.getElementById(e.target.id);
        while (element &&
               !isNaN(element.offsetLeft) &&
               !isNaN(element.offsetTop)) {
            offsetLeft += element.offsetLeft;
            offsetTop += element.offsetTop;
            element = element.offsetParent;
        }
        posx -= offsetLeft;
        posy -= offsetTop;
//        console.log('Event at (x='+posx.toString() +', y='+ posy.toString()+') in Element ' + e.target.id);
        return [posx, posy];
    },

  checkBrowser: function () {
      var ua = navigator.userAgent;
      var idx = ua.indexOf('Chrome/');
      if (idx === -1) {
          $("body").replaceWith("<body><p>Please use Google Chrome</p></body>");
          alert("Please Use Google Chrome");
          return false;
      }
      var vm = ua.slice(idx+7, idx+9);
      if (vm <= 30) {
          $("body").replaceWith("<body><p>Please update your Chrome Browser.</p></body>");
          alert("Please update your Chrome Browser.");
          return false;
      }
      return true;
  },

  init :  function () {
 //   if (!this.checkBrowser()) {return}
    this.ROBOT = window.location.host.split(':')[0];//Use localhost when serving website directly from robot 
    this.PORT = '9091';//Must match port on which rosbridge is being served
    $("body").css({"height": window.innerHeight, "width":window.innerWidth})
             .on('dragstart', function (event) { return false;})
             .on('drop', function (event) { return false;});
    window.addEventListener('resize', function () {
        $("body").css({"height": window.innerHeight, "width":window.innerWidth});
    });
    initUserLog('#notifications');

  //  $('#tabs').css({'top':'0px'})
  //  var tabs = $("#tabs").tabs();
  //  $('#cont_sel_container').buttonset();
  //  $('label:first','#cont_sel_container').removeClass('ui-corner-left')
  //                                        .addClass('ui-corner-top centered');
  //  $('label:last','#cont_sel_container').removeClass('ui-corner-right')
  //                                       .addClass('ui-corner-bottom centered');
  //  $('#scale_slider').slider({value:0.5, min:0, max:1.0,
  //                             step:0.01, orientation:'vertical'}); 
  //  $('.bpd, #cart_controller, .ar_servo_button, .traj_play_cont,'+
  //    '#adj_mirror, #traj_play_reverse, #ell_controller, #reg_head,'+
  //    '#rezero_wrench, #send_shave_select, #shave, #shave_stop, #tool_power').button();
    var grayVideo = function () {
        var w = $('body').width();
        var h = $('body').height();
        $('#image-cover').css({'height':h, 'width':w}).text("Connection Lost").addClass('connection-lost').show();

    };

    var clearVideo = function () {
        $('#image-cover').hide();
    };

    this.ros = new ROSLIB.Ros({url: 'ws://'+ this.ROBOT + ':'+ this.PORT});
    this.ros.on('close', function(e) {
        grayVideo();
        console.log("Disconnected or Can't Connect to " + this.ROBOT + ":"+ this.PORT + ".");
      }
    );
    this.ros.on('error', function(e) {
        grayVideo();
      log("Rosbridge Connection Error!");
      }
    );

    this.ros.on('connection', function(e) {
        console.log("Connected to " + RFH.ROBOT + ".");
        clearVideo();
        extendROSJS(RFH.ros);
//        RFH.connectionMonitor = new RFH.ConnectionMonitor({divId: 'network-status'}).start();
        RFH.batteryMonitor = new RFH.BatteryMonitor({ros: RFH.ros,
                                                     div: 'battery-status'});
        RFH.tfClient = new ROSLIB.TFClient({ros : RFH.ros,
                                            angularThres : 0.001,
                                            transThres : 0.001,
                                            rate : 10.0,
                                            fixedFrame : '/base_link' });
        RFH.tfClient.actionClient.cancel();
        RFH.pr2 = new PR2(RFH.ros);
        RFH.runStop = new RFH.RunStop({ros: RFH.ros});
        initMjpegCanvas();
        initViewer('video-main');
        RFH.rightEEDisplay = new RFH.EEDisplay({side:'r',
                                             ros: RFH.ros,
                                             tfClient: RFH.tfClient});
        RFH.leftEEDisplay = new RFH.EEDisplay({side: 'l',
                                             ros: RFH.ros,
                                             tfClient: RFH.tfClient});
        RFH.initTaskMenu('main-menu');
        RFH.smach = new RFH.Smach({displayContainer: $('#smach-container'),
                                   ros: RFH.ros});
        RFH.undo = new RFH.Undo({ros: RFH.ros,
                                 undoTopic: '/undo',
                                 buttonDiv: 'undo',
                                 rightEEDisplay: RFH.rightEEDisplay,
                                 leftEEDisplay: RFH.leftEEDisplay});
        RFH.kinectHeadPointCloud = new RFH.PointCloudView({ros: RFH.ros,
                                                           topic: "/pcl_filters/peek_points",
                                                           maxPoints: 16000,
                                                           tfClient: RFH.tfClient });
//        RFH.dataLogger = new RFH.DataLogger({ros: RFH.ros, topic: "/interface_log"});
//        initClickableActions();
//        initPr2(); 
//        initGripper('horizontal');
//        initTorsoSlider('horizontal');
//        initTTS('tabTTS');
//        initARServoTab('tabServoing');
//        initTrajPlay();
//        initBodyRegistration('tabBodyReg');
//        initEllControl();
//        initCartControl();
//        initTaskInterface('tabTasks');
//        initRYDSTab('tabRYDS');
//        if (window.location.hash.search('ft') !== -1) {
//          initFTDisplay('FTDisplay', {});
//        }
//        if (window.location.hash.search('skin') !== -1) {
//          initSkinUtils();
//        }
//        teleopHead();
      }
    );
  }
};

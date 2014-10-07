var RFH = {
    positionInElement: function (e) {
        var posx = 0;
        var posy = 0;
        if (!e) var e = window.event;
        if (e.pageX || e.pageY) 	{
            posx = e.pageX;
            posy = e.pageY;
        }
        else if (e.clientX || e.clientY) 	{
            posx = e.clientX + document.body.scrollLeft
                + document.documentElement.scrollLeft;
            posy = e.clientY + document.body.scrollTop
                + document.documentElement.scrollTop;
        }
        var offsetLeft = 0;
        var offsetTop = 0;
        var element = document.getElementById(e.target.id);
        while (element && !isNaN(element.offsetLeft)
                && !isNaN(element.offsetTop)) {
            offsetLeft += element.offsetLeft;
            offsetTop += element.offsetTop;
            element = element.offsetParent;
        }
        posx -= offsetLeft;
        posy -= offsetTop;
        console.log('Event at (x='+posx.toString() +', y='+ posy.toString()+') in Element ' + e.target.id);
        return [posx, posy]
    },

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

  init :  function () {
    if (!this.checkBrowser()) {return}

    this.ROBOT = window.location.host.split(':')[0];//Use localhost when serving website directly from robot 
    this.PORT = '9091';//Must match port on which rosbridge is being served
    $("body").css({"height": window.innerHeight, "width":window.innerWidth});
    window.addEventListener('resize', function () {
        $("body").css({"height": window.innerHeight, "width":window.innerWidth});
    });
    initUserLog('#notifications')

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

    this.ros = new ROSLIB.Ros({url: 'ws://'+ this.ROBOT + ':'+ this.PORT});
    this.ros.on('close', function(e) {
        log("Disconnected or Can't Connect to " + this.ROBOT + ":"+ this.PORT + ".");
      }
    );
    this.ros.on('error', function(e) {
      log("Rosbridge Connection Error!");
      }
    );
    this.ros.on('connection', function(e) {
        log("Connected to " + RFH.ROBOT + ".");
        extendROSJS(RFH.ros);
        initMjpegCanvas('video-main');
        initMarkerDisplay('markers');
        RFH.initTaskMenu('main-menu');
//        initClickableActions();
//        initPr2(); 
//        initGripper('horizontal');
//        initTorsoSlider('horizontal');
//        initTTS('tabTTS');
//        initARServoTab('tabServoing');
//        initRunStop('runStopDiv');
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

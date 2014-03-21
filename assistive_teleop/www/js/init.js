var assistive_teleop = {
    start :  function () {
        window.ros = new ROS('ws://'+ ROBOT + ':'+ PORT);
            ros.on('close', function(e) {
              log("Disconnected or Can't Connect to " + ROBOT + ":"+ PORT + ".");
            });
            ros.on('error', function(e) {
              log("Rosbridge Connection Error!");
            });
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
              $('#tabs').css({'top':'0px'})
              var tabs = $("#tabs").tabs();
              tabs.find(".ui-tabs-nav").sortable({axis:"xy", stop: function() {tabs.tabs("refresh");}});
            });
    }
};

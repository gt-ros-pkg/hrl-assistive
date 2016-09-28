var RFH = (function (module) {
    module.positionInElement = function (e) {
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
    };

    var checkBrowser = function () {
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
    };

    module.init = function () {
        //   if (!checkBrowser()) {return}
        this.ROBOT = window.location.host.split(':')[0];//Use localhost when serving website directly from robot 
        this.PORT = '9091';//Must match port on which rosbridge is being served
        $("body").css({"height": window.innerHeight, "width":window.innerWidth})
            .on('dragstart', function (event) { return false;})
            .on('drop', function (event) { return false;});
        window.addEventListener('resize', function () {
            $("body").css({"height": window.innerHeight, "width":window.innerWidth});
        });

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
            console.log("Rosbridge Connection Error!");
        }
        );

        this.ros.on('connection', function(e) {
            console.log("Connected to " + RFH.ROBOT + ".");
            clearVideo();
            RFH.extendROSJS(RFH.ros);
            //        RFH.connectionMonitor = new RFH.ConnectionMonitor({divId: 'network-status'}).start();
            RFH.batteryMonitor = new RFH.BatteryMonitor({ros: RFH.ros,
                div: 'battery-status'});
            RFH.tfClient = new ROSLIB.TFClient({ros : RFH.ros,
                angularThres : 0.001,
                transThres : 0.001,
                rate : 10.0,
                fixedFrame : '/base_link' });
            RFH.tfClient.actionClient.cancel();
            RFH.pr2 = new RFH.PR2(RFH.ros);
            RFH.runStop = new RFH.RunStop({ros: RFH.ros});

            RFH.mjpeg = new RFH.MjpegClient({ros: RFH.ros,
                imageTopic: '/head_mount_kinect/hd/image_color',
                infoTopic: '/head_mount_kinect/hd/camera_info',
                divId: 'video-main',
                imageId: 'mjpeg-image',
                host: RFH.ROBOT,
                port: 8080,
                quality: 75,
                tfClient:RFH.tfClient});
            RFH.mjpeg.cameraModel.infoSubCBList.push(RFH.mjpeg.refreshSize);
            RFH.mjpeg.cameraModel.updateCameraInfo();

            RFH.initViewer('video-main');
            RFH.initSkin();
            RFH.rightEEDisplay = new RFH.EEDisplay({side:'r',
                ros: RFH.ros,
                tfClient: RFH.tfClient});
            RFH.leftEEDisplay = new RFH.EEDisplay({side: 'l',
                ros: RFH.ros,
                tfClient: RFH.tfClient});

            RFH.leftSkinDisplay = new RFH.SkinDisplay({tfClient: RFH.tfClient,
                                                          head: RFH.pr2.head,
                                                          camera: RFH.mjpeg.cameraModel,
                                                          skins: [RFH.skins.left.upperarm,
                                                                  RFH.skins.left.forearm]
            });
            RFH.rightSkinDisplay = new RFH.SkinDisplay({tfClient: RFH.tfClient,
                                                          head: RFH.pr2.head,
                                                          camera: RFH.mjpeg.cameraModel,
                                                          skins: [RFH.skins.right.upperarm,
                                                                  RFH.skins.right.forearm]
            });
            RFH.initActionMenu('main-menu');
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
            RFH.dataLogger = new RFH.DataLogger({ros: RFH.ros, topic: "/interface_log"});
            RFH.heartbeatMonitor = new RFH.HeartbeatMonitor({ros: RFH.ros});
            RFH.shaver = new RFH.ShaverToggle({ros: RFH.ros, divId:'toggle-shaver-button'});
            RFH.initTaskMenu();

            /* Added content for left column */
            var showLeftColumn = function (event) {
                $('#left-col').animate({'left':0}, {duration:200, easing:'easeOutCubic'});
            };
            var hideLeftColumn = function (event) {
                if (event !== undefined) {
                    var toEl = $(event.toElement);
                    if (toEl.hasClass('ui-menu-item') || toEl.hasClass('ui-menu')) { return ; }
                }
                $('#left-col').animate({'left':'-200px'}, {duration: 350, easing:'easeInCubic'});
            };

            $('#left-col-small').on('mouseenter.rfh', showLeftColumn);
            $('#left-col').on('mouseleave.rfh blur.rfh', hideLeftColumn);
            setTimeout(hideLeftColumn, 2500);

            $('.left-col-accordion').accordion({collapsible: true, heightStyle: 'content', animate:75});
            //$('.task-menu-accordion').accordion('option', 'active', false);

        });
    };
    return module;
})(RFH || {});


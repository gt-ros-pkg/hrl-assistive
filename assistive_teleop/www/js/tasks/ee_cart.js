RFH.CartesianEEControl = function (options) {
    'use strict';
    var self = this;
    self.div = options.div || 'mjpeg';
    self.arm = options.arm;
    self.side = self.arm.side[0];
    self.gripper = options.gripper;
    self.smooth = self.arm instanceof PR2ArmJTTask;
    self.tfClient = options.tfClient;
    self.camera = options.camera;
    self.buttonText = self.side === 'r' ? 'Right_Hand' : 'Left_Hand';
    self.buttonClass = 'hand-button';
    $('#touchspot-toggle').button()
    $('#touchspot-toggle-label').hide();

    /// POSITION CONTROLS ///
    self.posCtrlId = self.side+'posCtrlIcon';
    self.targetIcon = new RFH.EECartControlIcon({divId: self.posCtrlId,
                                                 parentId: self.div,
                                                 arm: self.arm,
                                                 smooth: self.smooth});
    var handCtrlCSS = {bottom:"6%"};
    handCtrlCSS[self.arm.side] = "7%";
    $('#'+self.posCtrlId).css(handCtrlCSS).hide();

    /// ROTATION CONTROLS ///
    self.rotCtrlId = self.side+'rotCtrlIcon';
    self.rotIcon = new RFH.EERotControlIcon({divId: self.rotCtrlId,
                                                parentId: self.div,
                                                 arm: self.arm,
                                                 smooth: self.smooth});
    $('#'+self.rotCtrlId).css(handCtrlCSS).hide();

    /// SWITCH POSITION AND ROTATION ///
    $('#'+self.side+'-posrot-set').buttonset().hide().on('change.rfh', function (event, ui) {
            var mode = event.target.id.slice(-3);
            $('#'+self.side+'posCtrlIcon, #'+self.side+'rotCtrlIcon').hide();
            $('#'+self.side+mode+'CtrlIcon').show();
        });
//    $('#'+self.side+'-posrot-pos').click();

    /// TRACKING HAND WITH CAMERA ///
    self.updateTrackHand = function (event) {
        if ( $("#"+self.side+"-track-hand-toggle").is(":checked") ){
            self.trackHand();
        } else {
            clearInterval(RFH.pr2.head.pubInterval);
        }
    }

    self.trackHand = function () {
        clearInterval(RFH.pr2.head.pubInterval);
        RFH.pr2.head.pubInterval = setInterval(function () {
            RFH.pr2.head.pointHead(0, 0, 0, self.side+'_gripper_tool_frame');
        }, 100);
    }
    $("#"+self.side+"-track-hand-toggle").button().on('change.rfh', self.updateTrackHand);
    $("#"+self.side+"-track-hand-toggle-label").hide();


    /// GRIPPER SLIDER CONTROLS ///
    self.gripperDisplayDiv = self.side+'GripperDisplay';
    self.gripperDisplay = new RFH.GripperDisplay({gripper: self.gripper,
                                                   parentId: self.div,
                                                   divId: self.gripperDisplayDiv});
    var gripperCSS = {position: "absolute",
                      height: "3%",
                      width: "25%",
                      bottom: "2%"};
    gripperCSS[self.arm.side] = "3%";
    $('#'+self.gripperDisplayDiv).css( gripperCSS ).hide();


    /// TASK START/STOP ROUTINES ///
    self.start = function () {
        $("#touchspot-toggle-label, #"+self.side+"-track-hand-toggle-label, #"+self.side+"-posrot-set").show();
        var mode = $('#'+self.side+'-posrot-set>input:checked').attr('id').slice(-3);
        $('#'+self.side+mode+'CtrlIcon').show();
        $("#"+self.gripperDisplayDiv).show();
        self.updateTrackHand();
    }
    
    self.stop = function () {
        $('#'+self.posCtrlId + ', #'+self.rotCtrlId+', #touchspot-toggle-label, #'+self.side+'-track-hand-toggle-label, #'+self.side+'-posrot-set').hide();
        clearInterval(RFH.pr2.head.pubInterval);
        $('#'+self.gripperDisplayDiv).hide();
    };
}

RFH.EECartControlIcon = function (options) {
    'use strict';
    var self = this;
    self.divId = options.divId;
    self.parentId = options.parentId;
    self.arm = options.arm;
    self.lastDragTime = new Date();
    self.container = $('<div/>', {id: self.divId,
                                  class: "cart-ctrl-container"}).appendTo('#'+self.parentId);
    self.away = $('<div/>', {class: "away-button"}).appendTo('#'+self.divId).button();
    self.target = $('<div/>', {class: "target-trans"}).appendTo('#'+self.divId);
    self.toward = $('<div/>', {class: "toward-button"}).appendTo('#'+self.divId).button();
    $('#'+self.divId+' .target-trans').draggable({containment:"parent",
                                 distance: 8,
                                 revertDuration: 100,
                                 revert: true})
                                 .on("dragstart", function (event) { event.stopPropagation() });

    self.awayCB = function (event) {
        var dx = self.smooth ? 0.005 : 0.03;
        var dt = self.smooth ? 50 : 1000;
        if ($('#'+self.divId+' .away-button').hasClass('ui-state-active')) {
            var goal = self.arm.ros.composeMsg('geometry_msgs/PoseStamped');
            goal.header.frame_id = '/torso_lift_link';
            goal.pose.position = self.arm.state.pose.position;
            goal.pose.position.x += dx;
            goal.pose.orientation = self.arm.state.pose.orientation;
            self.arm.goalPosePublisher.publish(goal);
            setTimeout(function () {self.awayCB(event)}, dt);
        } 
    }
    $('#'+self.divId+' .away-button').on('mousedown.rfh', self.awayCB);

    self.towardCB = function (event) {
        var dx = self.smooth ? 0.003 : 0.03;
        var dt = self.smooth ? 100 : 1000;
        if ($('#'+self.divId+' .toward-button').hasClass('ui-state-active')){
            var goal = self.arm.ros.composeMsg('geometry_msgs/PoseStamped');
            goal.header.frame_id = '/torso_lift_link';
            goal.pose.position = self.arm.state.pose.position;
            goal.pose.position.x -= dx;
            goal.pose.orientation = self.arm.state.pose.orientation;
            self.arm.goalPosePublisher.publish(goal);
            setTimeout(function () {self.towardCB(event)}, dt);
        }
    }
    $('#'+self.divId+' .toward-button').on('mousedown.rfh', self.towardCB);

    self.onDrag = function (event, ui) {
        var mod_del = self.smooth ? 0.005 : 0.0025;
        var dt = self.smooth ? 100 : 1000;
        clearTimeout(self.dragTimer);
        var time = new Date();
        var timeleft = time - self.lastDragTime;
        if (timeleft > 1000) {
            self.lastDragTime = time;
            var dx = -mod_del * (ui.position.left - ui.originalPosition.left);
            var dy = -mod_del * (ui.position.top - ui.originalPosition.top);
            var goal = self.arm.ros.composeMsg('geometry_msgs/PoseStamped');
            goal.header.frame_id = '/torso_lift_link';
            goal.pose.position = self.arm.state.pose.position;
            goal.pose.position.y += dx;
            goal.pose.position.z += dy;
            goal.pose.orientation = self.arm.state.pose.orientation;
            self.arm.goalPosePublisher.publish(goal);
            self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, dt);
        } else {
            self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, timeleft);
        }

    }

    self.dragStop = function (event, ui) {
        clearTimeout(self.dragTimer);
    }
    $('#'+self.divId+' .target-trans').on('drag', self.onDrag).on('dragstop', self.dragStop);
}

RFH.EERotControlIcon = function (options) {
    'use strict';
    var self = this;
    self.divId = options.divId;
    self.parentId = options.parentId;
    self.arm = options.arm;
    self.lastDragTime = new Date();
    self.container = $('<div/>', {id: self.divId,
                                  class: "cart-ctrl-container"}).appendTo('#'+self.parentId);
    self.cwRot = $('<div/>', {class: "cw-button"}).appendTo('#'+self.divId).button();
    self.target = $('<div/>', {class: "target-rot"}).appendTo('#'+self.divId);
    self.ccwRot = $('<div/>', {class: "ccw-button"}).appendTo('#'+self.divId).button();
    $('#'+self.divId+' .target-rot').on('dragstart', function(event) { event.stopPropagation()})
                                    .draggable({containment:"parent",
                                                distance: 8,
                                                revertDuration: 100,
                                                revert: true});
    self.rpy_to_quat = function (roll, pitch, yaw) {
        // Convert from RPY
        var phi = roll / 2.0;
        var the = pitch / 2.0;
        var psi = yaw / 2.0;
        var x = Math.sin(phi) * Math.cos(the) * Math.cos(psi) - 
                Math.cos(phi) * Math.sin(the) * Math.sin(psi);
        var y = Math.cos(phi) * Math.sin(the) * Math.cos(psi) + 
                Math.sin(phi) * Math.cos(the) * Math.sin(psi);
        var z = Math.cos(phi) * Math.cos(the) * Math.sin(psi) - 
                Math.sin(phi) * Math.sin(the) * Math.cos(psi);
        var w = Math.cos(phi) * Math.cos(the) * Math.cos(psi) + 
                Math.sin(phi) * Math.sin(the) * Math.sin(psi);
        var quaternion = new ROSLIB.Quaternion({x:x, y:y, z:z, w:w});
        quaternion.normalize();
        return quaternion;
        }

    self.ccwCB = function (event) {
        var dAng= self.smooth ? Math.PI/100 : Math.PI/20;
        var dt = self.smooth ? 50 : 1000;
        if ($('#'+self.divId+' .ccw-button').hasClass('ui-state-active')) {
            var goal = self.arm.ros.composeMsg('geometry_msgs/PoseStamped');
            goal.header.frame_id = '/torso_lift_link';
            goal.pose = self.arm.state.pose;
            var dQuat = self.rpy_to_quat(-dAng, 0, 0);
            var quat = new ROSLIB.Quaternion(goal.pose.orientation);
            quat.multiply(dQuat)
            quat.normalize();
            goal.pose.orientation = quat;
            self.arm.goalPosePublisher.publish(goal);
            setTimeout(function () {self.ccwCB(event)}, dt);
        } 
    }
    $('#'+self.divId+' .ccw-button').on('mousedown.rfh', self.ccwCB);

    self.cwCB = function (event) {
        var dAng = self.smooth ? Math.PI/100 : Math.PI/20;
        var dt = self.smooth ? 100 : 1000;
        if ($('#'+self.divId+' .cw-button').hasClass('ui-state-active')){
            var goal = self.arm.ros.composeMsg('geometry_msgs/PoseStamped');
            goal.header.frame_id = '/torso_lift_link';
            goal.pose = self.arm.state.pose;
            var dQuat = self.rpy_to_quat(dAng, 0, 0);
            var quat = new ROSLIB.Quaternion(goal.pose.orientation);
            quat.multiply(dQuat)
            quat.normalize();
            goal.pose.orientation = quat;
            self.arm.goalPosePublisher.publish(goal);
            setTimeout(function () {self.cwCB(event)}, dt);
        }
    }
    $('#'+self.divId+' .cw-button').on('mousedown.rfh', self.cwCB);

    self.onDrag = function (event, ui) {
        // x -> rot around Z
        // y -> rot around y
        var dAng = self.smooth ? Math.PI/100 : Math.PI/20;
        var dt = self.smooth ? 100 : 1000;
        clearTimeout(self.dragTimer);
        var time = new Date();
        var timeleft = time - self.lastDragTime;
        if (timeleft > 1000) {
            self.lastDragTime = time;
            var dx = -mod_del * (ui.position.left - ui.originalPosition.left);
            var dy = -mod_del * (ui.position.top - ui.originalPosition.top);
            var goal = self.arm.ros.composeMsg('geometry_msgs/PoseStamped');
            goal.header.frame_id = '/torso_lift_link';
            goal.pose.position = self.arm.state.pose.position;
            goal.pose.position.y += dx;
            goal.pose.position.z += dy;
            goal.pose.orientation = self.arm.state.pose.orientation;
            self.arm.goalPosePublisher.publish(goal);
            self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, dt);
        } else {
            self.dragTimer = setTimeout(function () {self.onDrag(event, ui)}, timeleft);
        }

    }

    self.dragStop = function (event, ui) {
        clearTimeout(self.dragTimer);
    }
    $('#'+self.divId+' .target-rot').on('drag', self.onDrag).on('dragstop', self.dragStop);
}

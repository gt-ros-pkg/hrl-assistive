
var PR2 = function(){
    var pr2 = this;
    options = options || {};
    pr2.head = new pr2Head();
    pr2.tosro = new pr2Torso();
    pr2.base = new pr2Base();
    pr2.grippers = {'right': new pr2Gripper('right'),
                    'left': new pr2GRipper('left')}
    pr2.arms = {'right': new pr2Arm('right'),
                'left': new pr2Arm('left')}
}


var pr2Head = function(){
    var head = this;
    head.state = [0.0, 0.0];
    head.joints = ['',''];
    getMsgDetails('pr2_controllers_msgs/JointTrajectoryActionGoal')
    head.joint_pub = new window.ros.Topic({
                        name:'head_traj_controller/joint_trajectory_action/goal',
                        messageType:'pr2_controllers_msgs/JointTrajectoryActionGoal'});
    head.joint_pub.advertise();

    getMsgDetails('pr2_controllers_msgs/PointHeadActionGoal')
    head.point_pub = new window.ros.Topic({
                        name:'head_traj_controller/point_head_action/goal',
                        messageType:'pr2_controllers_msgs/PointHeadActionGoal'});
    head.point_pub.advertise();

    head.state_sub = new window.ros.Topic({
                        name:'/head_traj_controller/state_throttled',
                        messageType:'pr2_controllers_msgs/JointTrajectoryControllerState'});
    head.state_sub.subscribe(function(msg){
                                head.state = msg.actual.positions
                            });

    head.setAngles = function(pan, tilt){};
    head.stepAngles = function(delPan, delTilt){};
    head.pointHead = function(x,y,z,frame){};
};


var pr2Torso = function(){
    var torso = this;
    torso.state = 0.0;
    getMsgDetails('pr2_controllers_msgs/SingleJointPositionActionGoal')
    
    torso.goal_pub = new window.ros.Topic({
                        name:'torso_controller/position_joint_action/goal',
                        messageType:'pr2_controllers_msgs/SingleJointPositionActionGoal'});
    torso.goal_pub.advertise();

    torso.state_sub = new window.ros.Topic({
                        name: 'torso_controller/state_throttled',
                        messageType: 'pr2_controllers_msgs/JointTrajectoryControllerState'
                        });
    torso.state_sub.subscribe(function(msg){
                                torso.state = msg.actual.positions[0];
                              });

    torso.setPosition = function(z){
        var dir = (z < torso.state) ? 'Lowering' : 'Raising';
        log(dir+" Torso");
        console.log('Commanding torso'+
                    ' from z='+torso.state.toString()+
                    ' to z='+z.toString());
        var goal_msg = composeMsg('pr2_controllers_msgs/SingleJointPositionActionGoal');
        goal_msg.goal.position = z;
        goal_msg.goal.max_velocity = 1.0;
        var msg = new window.ros.Message(goal_msg);
        torso.goal_pub.publish(msg);
    };
};

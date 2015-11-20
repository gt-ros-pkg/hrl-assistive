#include "hrl_manipulation_task/robot.h"

Robot::Robot(ros::NodeHandle &nh, std::string base_link, std::string target_link):
  nh_(nh)
{
    ROS_INFO("Initialize robot main()");

    radius_ = 0.03;
    
    base_link_ = base_link;
    target_link_ = target_link;

    getParams();
    setupKDL();

    ROS_INFO("Initialize robot complete");
}

Robot::~Robot()
{
    std::vector<std::string> T;
    left_arm_joint_names_.swap(T);
    left_arm_link_names_.swap(T);
}


// Get the transformation matrix to the given link's frame based on the joint angles.
bool Robot::forwardKinematics(std::vector<double> &joint_angles, KDL::Frame &cartpos, int link_number)  const
{ 
    // Create solver based on kinematic chain
    KDL::ChainFkSolverPos_recursive fksolver = KDL::ChainFkSolverPos_recursive(ee_chain_);
   
    // Create joint array
    unsigned int nj = chain_.getNrOfJoints();
    if (static_cast<unsigned int>(joint_angles.size()) < nj)
    {
        std::cerr<<"Wrong number of joints in FK\n"; 
        return false;    
    }
  
    KDL::JntArray jointpositions(nj); 
    for (unsigned int i = 0; i < nj; i++)
    {
        jointpositions(i) = joint_angles[i];
    }
    
    // Calculate forward position kinematics
    bool kinematics_status;
    kinematics_status = fksolver.JntToCart(jointpositions, cartpos);
    // if(kinematics_status>=0){
    //     std::cout << cartpos <<std::endl;
    //     printf("%s \n","Success, thanks KDL!");
    // }else{
    if(kinematics_status<0)
    {
        ROS_ERROR("%s \n","Error: could not calculate forward kinematics :(");
    }
    return true;
}


bool Robot::getParams()
{
    std::string robot_desc_string;
    ros::param::get("robot_description", robot_desc_string);
    if (!kdl_parser::treeFromString(robot_desc_string, tree_)){
        ROS_ERROR("Failed to construct kdl tree");
        return false;
    }

    return true;
}

bool Robot::setupKDL()
{  
    ROS_INFO("Successfully setup KDL tree.");

    tree_.getChain(base_link_, "l_gripper_tool_frame", ee_chain_);
    tree_.getChain(base_link_, "l_wrist_flex_link", wrist_chain_);
    tree_.getChain(base_link_, "l_elbow_flex_link", elbow_chain_);
    tree_.getChain(base_link_, "l_upper_arm_link", shoulder_chain_);

    // std::cout << base_link_ << " " <<  target_link_ << std::endl;

    ROS_INFO("Successfully extracted arm KDL chain from complete tree.");
    // std::cout <<"Num segments " << chain_.getNrOfSegments() << ", Num joints " << chain_.getNrOfJoints() << std::endl;

    // XmlRpc::XmlRpcValue jt_min;
    // XmlRpc::XmlRpcValue jt_max;

    // std::string str_jt_min = "/haptic_mpc/";
    // std::string str_jt_max = "/haptic_mpc/";
        
    // // str_jt_min.append((string)robot_path);
    // // str_jt_max.append((string)robot_path);

    // str_jt_min.append("pr2/joint_limits/");
    // str_jt_max.append("pr2/joint_limits/");

    // str_jt_min.append("left");
    // str_jt_max.append("left");

    // str_jt_min.append("/min");
    // str_jt_max.append("/max");

    // while (nh_.getParam(str_jt_min, jt_min) == false)
    //     sleep(0.1);
    // while (nh_.getParam(str_jt_max, jt_max) == false)
    //     sleep(0.1);

    // // Number of joints check
    // assert((int)jt_min.size()==(int)chain_.getNrOfJoints()); 
    // assert((int)jt_max.size()==(int)chain_.getNrOfJoints());

    // qmin_.resize(chain_.getNrOfJoints());
    // qmax_.resize(chain_.getNrOfJoints());
        
    // // 
    // for (unsigned int i = 0; i < chain_.getNrOfJoints(); i++)
    // {
    //     qmin_(i) = (double((int)jt_min[i])-0.0)*M_PI/180.0; // added offset 0.3
    //     qmax_(i) = (double((int)jt_max[i])+0.0)*M_PI/180.0;
    // }

  return true;  
}

std::vector<std::string> Robot::getJointNames() const
{
    return left_arm_joint_names_;
}

bool Robot::setAllJoints(const sensor_msgs::JointState joint_state)
{
    // sensor_msgs::JointState joint_state;
    // joint_state = joint_state_monitor_->getJointStateRealJoints(); //joint_state_; // all joints

    if (joint_state.name.size() <=0){
        ROS_WARN("Joint State Monitor Failed. The monitor listens only joint_state. Probably joint_state or joint_states naming issue?");
        return false;
    }

    ROS_INFO("Got Joint All State from Monitor !!");
    for (unsigned int i = 0 ; i < joint_state.name.size(); ++i){
        // pm::KinematicState::JointState* js = state_->getJointState(joint_state.name[i]);
        // if(!js) continue;
        // js->setJointStateValues(joint_state.position[i]);
        joint_values_[joint_state.name[i]] = joint_state.position[i];
    }   

    // state_->setKinematicState(joint_values_);
    // state_->updateKinematicLinks();

    return true;
}





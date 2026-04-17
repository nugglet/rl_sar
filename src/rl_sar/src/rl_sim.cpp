/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "nlohmann/json.hpp"

// Constants
const int POS_SIZE = 3;
const int ROT_SIZE = 4;


// Quaternion operations
std::vector<double> quaternion_multiply(const std::vector<double>& q1, const std::vector<double>& q2) {
    std::vector<double> result(4);
    result[0] = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1];
    result[1] = q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0];
    result[2] = q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3];
    result[3] = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2];
    return result;
}

std::vector<double> quaternion_inverse(const std::vector<double>& q) {
    double norm = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    std::vector<double> result(4);
    result[0] = -q[0] / norm;
    result[1] = -q[1] / norm;
    result[2] = -q[2] / norm;
    result[3] = q[3] / norm;
    return result;
}

std::vector<double> quaternion_about_axis(double angle, const std::vector<double>& axis) {
    double half_angle = angle / 2.0;
    double sin_half = std::sin(half_angle);
    std::vector<double> result(4);
    result[0] = axis[0] * sin_half;
    result[1] = axis[1] * sin_half;
    result[2] = axis[2] * sin_half;
    result[3] = std::cos(half_angle);
    return result;
}

std::vector<double> quaternion_rotate_point(const std::vector<double>& point, const std::vector<double>& quat) {
    std::vector<double> q_point = {point[0], point[1], point[2], 0.0};
    std::vector<double> quat_inv = quaternion_inverse(quat);
    std::vector<double> temp = quaternion_multiply(quat, q_point);
    std::vector<double> rotated = quaternion_multiply(temp, quat_inv);
    return {rotated[0], rotated[1], rotated[2]};
}

double calc_heading(const std::vector<double>& q) {
    std::vector<double> ref_dir = {1.0, 0.0, 0.0};
    std::vector<double> rot_dir = quaternion_rotate_point(ref_dir, q);
    return std::atan2(rot_dir[1], rot_dir[0]);
}

std::vector<double> standardize_quaternion(std::vector<double> q) {
    if (q[3] < 0) {
        q[0] = -q[0];
        q[1] = -q[1];
        q[2] = -q[2];
        q[3] = -q[3];
    }
    return q;
}


RL_Sim::RL_Sim(int argc, char **argv)
{
#if defined(USE_ROS1)
    this->ang_vel_axis = "world";
    ros::NodeHandle nh;
    nh.param<std::string>("ros_namespace", this->ros_namespace, "");
    nh.param<std::string>("robot_name", this->robot_name, "");
#elif defined(USE_ROS2)
    ros2_node = std::make_shared<rclcpp::Node>("rl_sim_node");
    this->ang_vel_axis = "body";
    this->ros_namespace = ros2_node->get_namespace();
    // get params from param_node
    param_client = ros2_node->create_client<rcl_interfaces::srv::GetParameters>("/param_node/get_parameters");
    while (!param_client->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok()) {
            std::cout << LOGGER::ERROR << "Interrupted while waiting for param_node service. Exiting." << std::endl;
            return;
        }
        std::cout << LOGGER::WARNING << "Waiting for param_node service to be available..." << std::endl;
    }
    auto request = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
    request->names = {"robot_name", "gazebo_model_name"};
    // Use a timeout for the future
    auto future = param_client->async_send_request(request);
    auto status = rclcpp::spin_until_future_complete(ros2_node->get_node_base_interface(), future, std::chrono::seconds(5));
    if (status == rclcpp::FutureReturnCode::SUCCESS)
    {
        auto result = future.get();
        if (result->values.size() < 2)
        {
            std::cout << LOGGER::ERROR << "Failed to get all parameters from param_node" << std::endl;
        }
        else
        {
            this->robot_name = result->values[0].string_value;
            this->gazebo_model_name = result->values[1].string_value;
            std::cout << LOGGER::INFO << "Get param robot_name: " << this->robot_name << std::endl;
            std::cout << LOGGER::INFO << "Get param gazebo_model_name: " << this->gazebo_model_name << std::endl;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "Failed to call param_node service" << std::endl;
    }
#endif

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
#if defined(USE_ROS1)
    this->joint_publishers_commands.resize(this->params.Get<int>("num_of_dofs"));
#elif defined(USE_ROS2)
    this->robot_command_publisher_msg.motor_command.resize(this->params.Get<int>("num_of_dofs"));
    this->robot_state_subscriber_msg.motor_state.resize(this->params.Get<int>("num_of_dofs"));
#endif
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

#if defined(USE_ROS1)
    auto joint_controller_names_vec = this->params.Get<std::vector<std::string>>("joint_controller_names");  // avoid dangling reference
    this->StartJointController(this->ros_namespace, joint_controller_names_vec);
    // publisher
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/command";
        this->joint_publishers[joint_controller_name] =
            nh.advertise<robot_msgs::MotorCommand>(topic_name, 10);
    }

    // subscriber
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Sim::CmdvelCallback, this);
    this->joy_subscriber = nh.subscribe<sensor_msgs::Joy>("/joy", 10, &RL_Sim::JoyCallback, this);
    this->model_state_subscriber = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 10, &RL_Sim::ModelStatesCallback, this);
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/state";
        this->joint_subscribers[joint_controller_name] =
            nh.subscribe<robot_msgs::MotorState>(topic_name, 10,
                [this, joint_controller_name](const robot_msgs::MotorState::ConstPtr &msg)
                {
                    this->JointStatesCallback(msg, joint_controller_name);
                }
            );
        this->joint_positions[joint_controller_name] = 0.0f;
        this->joint_velocities[joint_controller_name] = 0.0f;
        this->joint_efforts[joint_controller_name] = 0.0f;
    }

    // service
    nh.param<std::string>("gazebo_model_name", this->gazebo_model_name, "");
    this->gazebo_pause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    this->gazebo_unpause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    this->gazebo_reset_world_client = nh.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
#elif defined(USE_ROS2)
    this->StartJointController(this->ros_namespace, this->params.Get<std::vector<std::string>>("joint_names"));
    // publisher
    this->robot_command_publisher = ros2_node->create_publisher<robot_msgs::msg::RobotCommand>(
        this->ros_namespace + "robot_joint_controller/command", rclcpp::SystemDefaultsQoS());

    // subscriber
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
    this->joy_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", rclcpp::SystemDefaultsQoS(),
        [this] (const sensor_msgs::msg::Joy::SharedPtr msg) {this->JoyCallback(msg);}
    );
    this->gazebo_imu_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", rclcpp::SystemDefaultsQoS(), [this] (const sensor_msgs::msg::Imu::SharedPtr msg) {this->GazeboImuCallback(msg);}
    );
    this->robot_state_subscriber = ros2_node->create_subscription<robot_msgs::msg::RobotState>(
        this->ros_namespace + "robot_joint_controller/state", rclcpp::SystemDefaultsQoS(),
        [this] (const robot_msgs::msg::RobotState::SharedPtr msg) {this->RobotStateCallback(msg);}
    );

    // service
    this->gazebo_pause_physics_client = ros2_node->create_client<std_srvs::srv::Empty>("/pause_physics");
    this->gazebo_unpause_physics_client = ros2_node->create_client<std_srvs::srv::Empty>("/unpause_physics");
    this->gazebo_reset_world_client = ros2_node->create_client<std_srvs::srv::Empty>("/reset_world");

    auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
    auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;
}

RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names)
{
#if defined(USE_ROS1)
    pid_t pid0 = fork();
    if (pid0 == 0)
    {
        std::string cmd = "rosrun controller_manager spawner joint_state_controller ";
        for (const auto& name : names)
        {
            cmd += name + " ";
        }
        cmd += "__ns:=" + ros_namespace;
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
#elif defined(USE_ROS2)
    const char* ros_distro = std::getenv("ROS_DISTRO");
    std::string spawner = (ros_distro && std::string(ros_distro) == "foxy") ? "spawner.py" : "spawner";

    std::filesystem::path tmp_path = std::filesystem::temp_directory_path() / "robot_joint_controller_params.yaml";
    {
        std::ofstream tmp_file(tmp_path);
        if (!tmp_file)
        {
            throw std::runtime_error("Failed to create temporary parameter file");
        }

        tmp_file << "/robot_joint_controller:\n";
        tmp_file << "    ros__parameters:\n";
        tmp_file << "        joints:\n";
        for (const auto& name : names)
        {
            tmp_file << "            - " << name << "\n";
        }
    }

    pid_t pid = fork();
    if (pid == 0)
    {
        std::string cmd = "ros2 run controller_manager " + spawner + " robot_joint_controller ";
        cmd += "-p " + tmp_path.string() + " ";
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
    else if (pid > 0)
    {
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
        {
            throw std::runtime_error("Failed to start joint controller");
        }

        std::filesystem::remove(tmp_path);
    }
    else
    {
        throw std::runtime_error("fork() failed");
    }
#endif
}


// ========================================== Motion Imitation Functions

    
void RL_Sim::load_motion(const std::string& filename) {
    this->_motion.load(filename);
}

std::vector<double> RL_Sim::build_target_obs(double time0, double dt, const std::vector<double>& sim_base_rot, const std::vector<double>& ref_base_pos) {
    std::vector<std::vector<double>> tar_poses;

    double heading = calc_heading(sim_base_rot);
    std::vector<double> inv_heading_rot = quaternion_about_axis(-heading, {0, 0, 1});

    for (int step : _tar_frame_steps) {
        double tar_time = time0 + step * dt;
        std::vector<double> tar_pose = _calc_ref_pose(tar_time);

        std::vector<double> tar_root_pos = _motion.get_frame_root_pos(tar_pose);
        std::vector<double> tar_root_rot = _motion.get_frame_root_rot(tar_pose);

        for (int i = 0; i < 3; ++i) {
            tar_root_pos[i] -= ref_base_pos[i];
        }
        tar_root_pos = quaternion_rotate_point(tar_root_pos, inv_heading_rot);

        tar_root_rot = quaternion_multiply(inv_heading_rot, tar_root_rot);
        tar_root_rot = standardize_quaternion(tar_root_rot);

        _motion.set_frame_root_pos(tar_root_pos, tar_pose);
        _motion.set_frame_root_rot(tar_root_rot, tar_pose);

        tar_poses.push_back(tar_pose);
    }

    std::vector<double> tar_obs;
    for (const auto& pose : tar_poses) {
        tar_obs.insert(tar_obs.end(), pose.begin(), pose.end());
    }

    return tar_obs;
}

std::vector<double> RL_Sim::_calc_ref_pose(double time) {
    std::vector<double> pose = _motion.calc_frame(time);

    // Apply origin offset
    std::vector<double> root_pos = _motion.get_frame_root_pos(pose);
    std::vector<double> root_rot = _motion.get_frame_root_rot(pose);

    root_rot = quaternion_multiply(_origin_offset_rot, root_rot);
    root_pos = quaternion_rotate_point(root_pos, _origin_offset_rot);
    for (int i = 0; i < 3; ++i) {
        root_pos[i] += _origin_offset_pos[i];
    }

    _motion.set_frame_root_rot(root_rot, pose);
    _motion.set_frame_root_pos(root_pos, pose);

    return pose;
}

// ============================== END Motion Imitation Functions 
void RL_Sim::GetState(RobotState<float> *state)
{
#if defined(USE_ROS1)
    const auto &orientation = this->pose.orientation;
    const auto &angular_velocity = this->vel.angular;
#elif defined(USE_ROS2)
    const auto &orientation = this->gazebo_imu.orientation;
    const auto &angular_velocity = this->gazebo_imu.angular_velocity;
#endif

    state->imu.quaternion[0] = orientation.w;
    state->imu.quaternion[1] = orientation.x;
    state->imu.quaternion[2] = orientation.y;
    state->imu.quaternion[3] = orientation.z;

    state->imu.gyroscope[0] = angular_velocity.x;
    state->imu.gyroscope[1] = angular_velocity.y;
    state->imu.gyroscope[2] = angular_velocity.z;

    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
#if defined(USE_ROS1)
        state->motor_state.q[i] = this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.dq[i] = this->joint_velocities[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
#elif defined(USE_ROS2)
        state->motor_state.q[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].q;
        state->motor_state.dq[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq;
        state->motor_state.tau_est[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;
#endif
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
#if defined(USE_ROS1)
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
#elif defined(USE_ROS2)
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
#endif
    }

#if defined(USE_ROS1)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->joint_publishers[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]].publish(this->joint_publishers_commands[i]);
    }
#elif defined(USE_ROS2)
    this->robot_command_publisher->publish(this->robot_command_publisher_msg);
#endif
}

void RL_Sim::RobotControl()
{
    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
#if defined(USE_ROS1)
        std_srvs::Empty empty;
        this->gazebo_reset_world_client.call(empty);
#elif defined(USE_ROS2)
        auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
        auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_pause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_pause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_unpause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_unpause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

#if defined(USE_ROS1)
void RL_Sim::ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg)
{
    this->vel = msg->twist[2];
    this->pose = msg->pose[2];
}
#elif defined(USE_ROS2)
void RL_Sim::GazeboImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    this->gazebo_imu = *msg;
}
#endif

void RL_Sim::CmdvelCallback(
#if defined(USE_ROS1)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}

void RL_Sim::JoyCallback(
#if defined(USE_ROS1)
    const sensor_msgs::Joy::ConstPtr &msg
#elif defined(USE_ROS2)
    const sensor_msgs::msg::Joy::SharedPtr msg
#endif
)
{
    this->joy_msg = *msg;

    // joystick control
    // Description of buttons and axes(F710):
    // |__ buttons[]: A=0, B=1, X=2, Y=3, LB=4, RB=5, back=6, start=7, power=8, stickL=9, stickR=10
    // |__ axes[]: Lx=0, Ly=1, Rx=3, Ry=4, LT=2, RT=5, DPadX=6, DPadY=7

    if (this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::A);
    if (this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::B);
    if (this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::X);
    if (this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->joy_msg.buttons[4]) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->joy_msg.buttons[5]) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[5]) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = this->joy_msg.axes[1]; // LY
    this->control.y = this->joy_msg.axes[0]; // LX
    this->control.yaw = this->joy_msg.axes[3]; // RX
}

#if defined(USE_ROS1)
void RL_Sim::JointStatesCallback(const robot_msgs::MotorState::ConstPtr &msg, const std::string &joint_controller_name)
{
    this->joint_positions[joint_controller_name] = msg->q;
    this->joint_velocities[joint_controller_name] = msg->dq;
    this->joint_efforts[joint_controller_name] = msg->tau_est;
}
#elif defined(USE_ROS2)
void RL_Sim::RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg)
{
    this->robot_state_subscriber_msg = *msg;
}
#endif

void RL_Sim::RunModel()
// Modify this to add in reading in and generating ref motions
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        this->ref_motions = this->load_motion("../policy/go2/motion_imitation/data/go2_run_config_11_pace.txt")
        if (this->control.navigation_mode)
        {
            this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Sim::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
    {
        // [ "dof_pos_abs", "dof_vel", "ang_vel", "gravity_vec", "actions"]
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        // TODO: add the 4 ref motions obs here
        this->history_obs = this->history_obs.insert(this->build_target_obs())

        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
#if defined(USE_ROS1)
        this->plot_real_joint_pos[i].push_back(this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]]);
        this->plot_target_joint_pos[i].push_back(this->joint_publishers_commands[i].q);
#elif defined(USE_ROS2)
        this->plot_real_joint_pos[i].push_back(this->robot_state_subscriber_msg.motor_state[i].q);
        this->plot_target_joint_pos[i].push_back(this->robot_command_publisher_msg.motor_command[i].q);
#endif
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

//======================================================================================================================

// std::vector<float> RL_Sim::load_motions(std::vector<std::string> filenames){
//     int num_files = filenames.size();

//     std::vector<float> motions;
    
//     for (std::string f : filenames) {
//         // parse the JSON file
//         std::ifstream file(f);
//         std::string content(std::istreambuf_iterator<char>{file}, 
//             std::istreambuf_iterator<char>{});
//         value jv = parse(content);

//         //extract data
//         auto motion = value_to<std::vector<std::vector<float>>>(jv.as_object()["Frames"]);
//         motions.push_back(motion);
//     }

//     return motions;
// };


// std::vector<float> RL_Sim::build_target_obs(){
//     // Internal function to construct sequence of target frames for future timesteps. Returns array of target frames

//     std::vector<std::vector<float>> tar_poses;

//     float time0 = this->episode_length_buf * this->params.Get<float>("dt") * this->params.Get<int>("decimation");
    
//     std::vector<float> motion = this->_get_active_motion();

//     std::vector<float> ref_base_pos = _get_ref_base_position();
//     std::vector<float> sim_base_rot = _get_base_orientation();

//     heading = _calc_heading(sim_base_rot);
//     inv_heading_rot = _quaternion_about_axis(-heading, [0, 0, 1]);

//     for (int& step : this->tar_frame_steps){

//         float tar_time = time0 + step * dt;
//         std::vector<float> tar_pose = this->_calc_ref_pose(tar_time);

//         std::vector<float> tar_root_pos = _get_frame_root_pos(tar_pose);
//         std::vector<float> tar_root_rot = _get_frame_root_rot(tar_pose);

//         tar_root_pos -= ref_base_pos;
//         tar_root_pos = _quaternion_rotate_point(tar_root_pos, inv_heading_rot);

//         tar_root_rot = _quaternion_multiply(inv_heading_rot, tar_root_rot);
//         tar_root_rot = _standardize_quaternion(tar_root_rot);

//         _set_frame_root_pos(tar_root_pos, tar_pose);
//         _set_frame_root_rot(tar_root_rot, tar_pose);

//         tar_poses.push_back(tar_pose);
//     }


//     std::vector<float> tar_obs.insert(tar_poses);

//     return tar_obs;

// };


// std::vector<float> RL_Sim::_get_active_motion(){
//     """Get index of the active reference motion currently being imitated.

//     Returns:
//     Index of the active reference motion.
//     """
//     return this->_ref_motions[this->_active_motion_id];

// };

// std::vector<float> _get_ref_base_position(){};

// std::vector<float> _get_base_orientation(){};

// std::vector<float> _calc_heading(std::vector<float> q){
//     """Returns the heading of a rotation q, specified as a quaternion.

//     The heading represents the rotational component of q along the vertical
//     axis (z axis).

//     Args:
//         q: A quaternion that the heading is to be computed from.

//     Returns:
//         An angle representing the rotation about the z axis.

//     """
//     std::vector<int> ref_dir = {1, 0, 0};
//     std::vector<float> rot_dir = _quaternion_rotate_point(ref_dir, q);
//     std::vector<float> heading = std::atan2(rot_dir[1], rot_dir[0]);
//     return heading;

// };

// std::vector<float> _quaternion_about_axis(std::vector<float> heading, std::vector<float> axis){

// };

// std::vector<float> RL_Sim::_calc_ref_pose(float time, bool apply_origin_offset=true){
//     """Calculates the reference pose for a given point in time.

//     Args:
//     time: Time elapsed since the start of the reference motion.
//     apply_origin_offset: A flag for enabling the origin offset to be applied
//         to the pose.

//     Returns:
//     An array containing the reference pose at the given point in time.
//     """
    
//     std::vector<float> pose(3);
//     std::vector<float> motion = this->_get_active_motion();
//     bool enable_warmup_pose = this->_curr_episode_warmup && time >= -this->_warmup_time && time < 0.0;
    
//     if (enable_warmup_pose){

//         pose = this->_calc_ref_pose_warmup();
//     }
    
//     else {
//         pose = this->_calc_frame(time);
//     };
    

//     // if (apply_origin_offset){
//     //     std::vector<float> root_pos = _get_frame_root_pos(pose);
//     //     std::vector<float> root_rot = _get_frame_root_rot(pose);

//     //     root_rot = _quaternion_multiply(this->_origin_offset_rot,
//     //                                                 root_rot)
//     //     root_pos = _quaternion_rotate_point(root_pos, self._origin_offset_rot)
//     //     root_pos += this->_origin_offset_pos

//     //     _set_frame_root_rot(root_rot, pose)
//     //     _set_frame_root_pos(root_pos, pose)
//     // };
    

//     return pose
// };

// std::vector<float> RL_Sim::_calc_frame(float time){
//     """Calculates the frame for a given point in time.

//     Args:
//       time: Time at which the frame is to be computed.
//     Return: An array containing the frame for the given point in time,
//       specifying the pose of the character.
//     """
//     f0, f1, blend = self.calc_blend_idx(time)

//     frame0 = self.get_frame(f0)
//     frame1 = self.get_frame(f1)
//     blend_frame = self.blend_frames(frame0, frame1, blend)

//     blend_root_pos = self.get_frame_root_pos(blend_frame)
//     blend_root_rot = self.get_frame_root_rot(blend_frame)

//     cycle_count = self.calc_cycle_count(time)
//     cycle_offset_pos = self._calc_cycle_offset_pos(cycle_count)
//     cycle_offset_rot = self._calc_cycle_offset_rot(cycle_count)

//     blend_root_pos = pose3d.QuaternionRotatePoint(blend_root_pos,
//                                                   cycle_offset_rot)
//     blend_root_pos += cycle_offset_pos

//     blend_root_rot = transformations.quaternion_multiply(
//         cycle_offset_rot, blend_root_rot)
//     blend_root_rot = motion_util.standardize_quaternion(blend_root_rot)

//     self.set_frame_root_pos(blend_root_pos, blend_frame)
//     self.set_frame_root_rot(blend_root_rot, blend_frame)

//     return blend_frame

// };

// std::vector<float> RL_Sim::_calc_ref_pose_warmup(){
//     """Calculate default reference  pose during warmup period."""
//     motion = this->_get_active_motion()
//     pose0 = motion.calc_frame(0)
//     warmup_pose = self._default_pose.copy()

//     pose_root_rot = motion.get_frame_root_rot(pose0)
//     default_root_rot = motion.get_frame_root_rot(warmup_pose)
//     default_root_pos = motion.get_frame_root_pos(warmup_pose)

//     pose_heading = motion_util.calc_heading(pose_root_rot)
//     default_heading = motion_util.calc_heading(default_root_rot)
//     delta_heading = pose_heading - default_heading
//     delta_heading_rot = transformations.quaternion_about_axis(
//         delta_heading, [0, 0, 1])

//     default_root_pos = pose3d.QuaternionRotatePoint(default_root_pos,
//                                                     delta_heading_rot)
//     default_root_rot = transformations.quaternion_multiply(
//         delta_heading_rot, default_root_rot)

//     motion.set_frame_root_pos(default_root_pos, warmup_pose)
//     motion.set_frame_root_rot(default_root_rot, warmup_pose)

//     return warmup_pose

// };

// std::vector<float> _get_frame_root_pos(std::vector<float> pose){};

// std::vector<float> _get_frame_root_rot(std::vector<float> pose){};

// std::vector<float> _quaternion_rotate_point(std::vector<float> point, std::vector<float> quat){
//     """Performs a rotation by quaternion.

//     Rotate the point by the quaternion using quaternion multiplication,
//     (q * p * q^-1), without constructing the rotation matrix.

//     Args:
//         point: The point to be rotated.
//         quat: The rotation represented as a quaternion [x, y, z, w].

//     Returns:
//         A 3D vector in a numpy array.
//     """

//     std::vector<float> q_point = {point[0], point[1], point[2], 0.0};
//     std::vector<float> quat_inverse = _quaternion_inverse(quat);
//     std::vector<float> x = _quaternion_multiply(quat, q_point);
//     std::vector<float> q_point_rotated = _quaternion_multiply(x, quat_inverse);
    
//     return q_point_rotated[:3];
// };

// std::vector<float> _quaternion_multiply(std::vector<float> quat_1, std::vector<float> quat_0){
//     """Return multiplication of two quaternions.

//     >>> q = quaternion_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
//     >>> numpy.allclose(q, [-44, -14, 48, 28])
//     True

//     """
//     float [x0, y0, z0, w0] = quat_0;
//     float [x1, y1, z1, w1] = quat_1;
//     std::vector<float> out = {x1*w0 + y1*z0 - z1*y0 + w1*x0,
//         -x1*z0 + y1*w0 + z1*x0 + w1*y0,
//         x1*y0 - y1*x0 + z1*w0 + w1*z0,
//         -x1*x0 - y1*y0 - z1*z0 + w1*w0};
//     return out;


// // ===================================================== Utils =============================================================

// // Quaternion operations
// std::vector<double> quaternion_multiply(const std::vector<double>& q1, const std::vector<double>& q2) {
//     std::vector<double> result(4);
//     result[0] = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1];
//     result[1] = q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0];
//     result[2] = q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3];
//     result[3] = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2];
//     return result;
// }

// std::vector<double> quaternion_inverse(const std::vector<double>& q) {
//     double norm = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
//     std::vector<double> result(4);
//     result[0] = -q[0] / norm;
//     result[1] = -q[1] / norm;
//     result[2] = -q[2] / norm;
//     result[3] = q[3] / norm;
//     return result;
// }

// std::vector<double> quaternion_about_axis(double angle, const std::vector<double>& axis) {
//     double half_angle = angle / 2.0;
//     double sin_half = std::sin(half_angle);
//     std::vector<double> result(4);
//     result[0] = axis[0] * sin_half;
//     result[1] = axis[1] * sin_half;
//     result[2] = axis[2] * sin_half;
//     result[3] = std::cos(half_angle);
//     return result;
// }

// std::vector<double> quaternion_rotate_point(const std::vector<double>& point, const std::vector<double>& quat) {
//     std::vector<double> q_point = {point[0], point[1], point[2], 0.0};
//     std::vector<double> quat_inv = quaternion_inverse(quat);
//     std::vector<double> temp = quaternion_multiply(quat, q_point);
//     std::vector<double> rotated = quaternion_multiply(temp, quat_inv);
//     return {rotated[0], rotated[1], rotated[2]};
// }

// double calc_heading(const std::vector<double>& q) {
//     std::vector<double> ref_dir = {1.0, 0.0, 0.0};
//     std::vector<double> rot_dir = quaternion_rotate_point(ref_dir, q);
//     return std::atan2(rot_dir[1], rot_dir[0]);
// }

// std::vector<double> standardize_quaternion(std::vector<double> q) {
//     if (q[3] < 0) {
//         q[0] = -q[0];
//         q[1] = -q[1];
//         q[2] = -q[2];
//         q[3] = -q[3];
//     }
//     return q;
// }

// // MotionData class
// class MotionData {
// public:
//     std::vector<std::vector<double>> _frames;
//     double _frame_duration;
//     int _loop_mode; // 0 for Clamp, 1 for Wrap

//     void load(const std::string& motion_file) {
//         std::ifstream f(motion_file);
//         nlohmann::json motion_json = nlohmann::json::parse(f);

//         _loop_mode = motion_json["LoopMode"] == "Wrap" ? 1 : 0;
//         _frame_duration = motion_json["FrameDuration"];
//         _frames = motion_json["Frames"].get<std::vector<std::vector<double>>>();

//         // Postprocess frames if needed
//         _postprocess_frames();
//     }

//     void _postprocess_frames() {
//         if (!_frames.empty()) {
//             std::vector<double> first_frame = _frames[0];
//             std::vector<double> pos_start = get_frame_root_pos(first_frame);

//             for (auto& frame : _frames) {
//                 std::vector<double> root_pos = get_frame_root_pos(frame);
//                 root_pos[0] -= pos_start[0];
//                 root_pos[1] -= pos_start[1];
//                 set_frame_root_pos(root_pos, frame);

//                 std::vector<double> root_rot = get_frame_root_rot(frame);
//                 // Normalize quaternion (assuming it's already normalized)
//                 root_rot = standardize_quaternion(root_rot);
//                 set_frame_root_rot(root_rot, frame);
//             }
//         }
//     }

//     std::vector<double> get_frame(int idx) {
//         return _frames[idx];
//     }

//     std::vector<double> calc_frame(double time) {
//         int f0, f1;
//         double blend;
//         calc_blend_idx(time, f0, f1, blend);

//         std::vector<double> frame0 = get_frame(f0);
//         std::vector<double> frame1 = get_frame(f1);
//         std::vector<double> blend_frame = blend_frames(frame0, frame1, blend);

//         // For simplicity, ignoring cycle offset for now
//         return blend_frame;
//     }

//     void calc_blend_idx(double time, int& f0, int& f1, double& blend) {
//         double duration = get_duration();
//         double num_frames = _frames.size();
//         double frame_time = time / _frame_duration;
//         f0 = static_cast<int>(std::floor(frame_time)) % static_cast<int>(num_frames);
//         f1 = (f0 + 1) % static_cast<int>(num_frames);
//         blend = frame_time - std::floor(frame_time);
//     }

//     std::vector<double> blend_frames(const std::vector<double>& frame0, const std::vector<double>& frame1, double blend) {
//         std::vector<double> result(frame0.size());
//         for (size_t i = 0; i < frame0.size(); ++i) {
//             result[i] = frame0[i] * (1.0 - blend) + frame1[i] * blend;
//         }
//         // Special handling for quaternions
//         size_t pos_end = POS_SIZE + ROT_SIZE;
//         for (size_t i = POS_SIZE; i < pos_end; ++i) {
//             // Slerp for quaternions, but for simplicity, linear blend
//             result[i] = frame0[i] * (1.0 - blend) + frame1[i] * blend;
//         }
//         return result;
//     }

//     double get_duration() {
//         return _frames.size() * _frame_duration;
//     }

//     std::vector<double> get_frame_root_pos(const std::vector<double>& frame) {
//         return {frame[0], frame[1], frame[2]};
//     }

//     void set_frame_root_pos(const std::vector<double>& root_pos, std::vector<double>& out_frame) {
//         out_frame[0] = root_pos[0];
//         out_frame[1] = root_pos[1];
//         out_frame[2] = root_pos[2];
//     }

//     std::vector<double> get_frame_root_rot(const std::vector<double>& frame) {
//         return {frame[3], frame[4], frame[5], frame[6]};
//     }

//     void set_frame_root_rot(const std::vector<double>& root_rot, std::vector<double>& out_frame) {
//         out_frame[3] = root_rot[0];
//         out_frame[4] = root_rot[1];
//         out_frame[5] = root_rot[2];
//         out_frame[6] = root_rot[3];
//     }
// };


#if defined(USE_ROS1)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
#if defined(USE_ROS1)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Sim rl_sar(argc, argv);
    ros::spin();
#elif defined(USE_ROS2)
    rclcpp::init(argc, argv);
    auto rl_sar = std::make_shared<RL_Sim>(argc, argv);
    rclcpp::spin(rl_sar->ros2_node);
    rclcpp::shutdown();
#endif
    return 0;
}


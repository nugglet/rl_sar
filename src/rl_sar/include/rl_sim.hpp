/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SIM_HPP
#define RL_SIM_HPP

// #define PLOT
// #define CSV_LOGGER

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_all.hpp"

#include <csignal>
#include <vector>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
// #include "library/nlohmann/json.hpp"
using json = nlohmann::json;

#if defined(USE_ROS1)
#include <ros/ros.h>
#include "std_srvs/Empty.h"
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>
#include <gazebo_msgs/ModelStates.h>
#include "robot_msgs/MotorCommand.h"
#include "robot_msgs/MotorState.h"
#elif defined(USE_ROS2)
#include "robot_msgs/msg/robot_command.hpp"
#include "robot_msgs/msg/robot_state.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_srvs/srv/empty.hpp>
#include <rcl_interfaces/srv/get_parameters.hpp>
#endif

#include "matplotlibcpp.h"


namespace plt = matplotlibcpp;

// Constants
const int POS_SIZE = 3;
const int ROT_SIZE = 4;


// MotionData class
class MotionData {
public:
    std::vector<std::vector<double>> _frames;
    double _frame_duration;
    int _loop_mode; // 0 for Clamp, 1 for Wrap

    void load(const std::string& motion_file) {
        std::ifstream f(motion_file);
        json::json motion_json = json::parse(f);

        _loop_mode = motion_json["LoopMode"] == "Wrap" ? 1 : 0;
        _frame_duration = motion_json["FrameDuration"];
        _frames = motion_json["Frames"].get<std::vector<std::vector<double>>>();

        // Postprocess frames if needed
        _postprocess_frames();
    }

    void _postprocess_frames() {
        if (!_frames.empty()) {
            std::vector<double> first_frame = _frames[0];
            std::vector<double> pos_start = get_frame_root_pos(first_frame);

            for (auto& frame : _frames) {
                std::vector<double> root_pos = get_frame_root_pos(frame);
                root_pos[0] -= pos_start[0];
                root_pos[1] -= pos_start[1];
                set_frame_root_pos(root_pos, frame);

                std::vector<double> root_rot = get_frame_root_rot(frame);
                // Normalize quaternion (assuming it's already normalized)
                root_rot = standardize_quaternion(root_rot);
                set_frame_root_rot(root_rot, frame);
            }
        }
    }

    std::vector<double> get_frame(int idx) {
        return _frames[idx];
    }

    std::vector<double> calc_frame(double time) {
        int f0, f1;
        double blend;
        calc_blend_idx(time, f0, f1, blend);

        std::vector<double> frame0 = get_frame(f0);
        std::vector<double> frame1 = get_frame(f1);
        std::vector<double> blend_frame = blend_frames(frame0, frame1, blend);

        // For simplicity, ignoring cycle offset for now
        return blend_frame;
    }

    void calc_blend_idx(double time, int& f0, int& f1, double& blend) {
        double duration = get_duration();
        double num_frames = _frames.size();
        double frame_time = time / _frame_duration;
        f0 = static_cast<int>(std::floor(frame_time)) % static_cast<int>(num_frames);
        f1 = (f0 + 1) % static_cast<int>(num_frames);
        blend = frame_time - std::floor(frame_time);
    }

    std::vector<double> blend_frames(const std::vector<double>& frame0, const std::vector<double>& frame1, double blend) {
        std::vector<double> result(frame0.size());
        for (size_t i = 0; i < frame0.size(); ++i) {
            result[i] = frame0[i] * (1.0 - blend) + frame1[i] * blend;
        }
        // Special handling for quaternions
        size_t pos_end = POS_SIZE + ROT_SIZE;
        for (size_t i = POS_SIZE; i < pos_end; ++i) {
            // Slerp for quaternions, but for simplicity, linear blend
            result[i] = frame0[i] * (1.0 - blend) + frame1[i] * blend;
        }
        return result;
    }

    double get_duration() {
        return _frames.size() * _frame_duration;
    }

    std::vector<double> get_frame_root_pos(const std::vector<double>& frame) {
        return {frame[0], frame[1], frame[2]};
    }

    void set_frame_root_pos(const std::vector<double>& root_pos, std::vector<double>& out_frame) {
        out_frame[0] = root_pos[0];
        out_frame[1] = root_pos[1];
        out_frame[2] = root_pos[2];
    }

    std::vector<double> get_frame_root_rot(const std::vector<double>& frame) {
        return {frame[3], frame[4], frame[5], frame[6]};
    }

    void set_frame_root_rot(const std::vector<double>& root_rot, std::vector<double>& out_frame) {
        out_frame[3] = root_rot[0];
        out_frame[4] = root_rot[1];
        out_frame[5] = root_rot[2];
        out_frame[6] = root_rot[3];
    }
};

class RL_Sim : public RL
{
public:
    RL_Sim(int argc, char **argv);
    ~RL_Sim();

    MotionData _motion;
    std::vector<int> _tar_frame_steps = {1, 2};
    std::vector<double> _origin_offset_rot = {0, 0, 0, 1};
    std::vector<double> _origin_offset_pos = {0, 0, 0};

    void load_motion(const std::string& filename);
    std::vector<double> build_target_obs(double time0, double dt, const std::vector<double>& sim_base_rot, const std::vector<double>& ref_base_pos);
  

#if defined(USE_ROS2)
    std::shared_ptr<rclcpp::Node> ros2_node;
#endif

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();
    std::vector<double> _calc_ref_pose(double time);

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // ros interface
    std::string ros_namespace;
#if defined(USE_ROS1)
    geometry_msgs::Twist vel;
    geometry_msgs::Pose pose;
    geometry_msgs::Twist cmd_vel;
    sensor_msgs::Joy joy_msg;
    ros::Subscriber model_state_subscriber;
    ros::Subscriber cmd_vel_subscriber;
    ros::Subscriber joy_subscriber;
    ros::ServiceClient gazebo_pause_physics_client;
    ros::ServiceClient gazebo_unpause_physics_client;
    ros::ServiceClient gazebo_reset_world_client;
    std::map<std::string, ros::Publisher> joint_publishers;
    std::map<std::string, ros::Subscriber> joint_subscribers;
    std::vector<robot_msgs::MotorCommand> joint_publishers_commands;
    void ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg);
    void JointStatesCallback(const robot_msgs::MotorState::ConstPtr &msg, const std::string &joint_controller_name);
    void CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg);
    void JoyCallback(const sensor_msgs::Joy::ConstPtr &msg);
#elif defined(USE_ROS2)
    sensor_msgs::msg::Imu gazebo_imu;
    geometry_msgs::msg::Twist cmd_vel;
    sensor_msgs::msg::Joy joy_msg;
    robot_msgs::msg::RobotCommand robot_command_publisher_msg;
    robot_msgs::msg::RobotState robot_state_subscriber_msg;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gazebo_imu_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscriber;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr gazebo_pause_physics_client;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr gazebo_unpause_physics_client;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr gazebo_reset_world_client;
    rclcpp::Publisher<robot_msgs::msg::RobotCommand>::SharedPtr robot_command_publisher;
    rclcpp::Subscription<robot_msgs::msg::RobotState>::SharedPtr robot_state_subscriber;
    rclcpp::Client<rcl_interfaces::srv::GetParameters>::SharedPtr param_client;
    void GazeboImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg);
    void JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);
#endif

    // others
    std::string gazebo_model_name;
    std::map<std::string, float> joint_positions;
    std::map<std::string, float> joint_velocities;
    std::map<std::string, float> joint_efforts;
    void StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names);
};

#endif // RL_SIM_HPP

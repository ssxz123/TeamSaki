#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <referee_pkg/srv/hit_armor.hpp>
#include <shooter_processor.hpp>
#include <vision_processor.hpp>
#include "opencv2/opencv.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

using namespace std;
using namespace cv;

class ShooterNode : public rclcpp::Node
{
    public:
        ShooterNode(string name) : Node(name)
        {
            RCLCPP_INFO(this->get_logger(), "This is %s", name.c_str());
            hit_armor_service = this->create_service<referee_pkg::srv::HitArmor>(
                "/referee/hit_arror",
                bind(&ShooterNode::call_back, this, placeholders::_1, placeholders::_2)
            );
            image_sub = this->create_subscription<sensor_msgs::msg::Image>(
                "/camera/image_raw",
                10,
                std::bind(&ShooterNode::camera_callback, this, std::placeholders::_1));
            
            
        }

    private:
        int stage;
        vector<Point2d>armor_points;
        rclcpp::Service<referee_pkg::srv::HitArmor>::SharedPtr hit_armor_service;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
        void camera_callback(sensor_msgs::msg::Image::SharedPtr msg);
        void call_back(
            const shared_ptr<referee_pkg::srv::HitArmor::Request> request,
            shared_ptr<referee_pkg::srv::HitArmor::Response> response);

        
};

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    auto node = make_shared<ShooterNode>("shooter_node");
    rclcpp::spin(node);
    rclcpp::shutdown();
}

void ShooterNode::camera_callback(sensor_msgs::msg::Image::SharedPtr msg){
    try{
        cv_bridge::CvImagePtr cv_ptr;
        cv::Mat img;
        if(msg->encoding == "rgb8" || msg->encoding == "R8G8B8"){
            cv_ptr = cv_bridge::toCvCopy(msg, "rgb8");
            cv::cvtColor(cv_ptr->image,img, cv::COLOR_RGB2BGR);
        }
        else{
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            img = cv_ptr->image;
        }

        if(img.empty()){
            RCLCPP_WARN(this->get_logger(), "Image Empty!");
            return;
        }
        // resize(img, img, Size(), 2, 2);

        imshow("Image", img);
        vector<Target> objects;
        VisionProcessor::Level2(img, objects);
        if(objects.size() == 0)return;
        armor_points.clear();
        for(int i = 0; i < 4; i++)
        {
            armor_points.push_back(objects[0].points[i]);
        }
        

    }
    catch(cv_bridge::Exception &e){
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void ShooterNode::call_back(const shared_ptr<referee_pkg::srv::HitArmor::Request> request,
    shared_ptr<referee_pkg::srv::HitArmor::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "已处理服务");

    // stage = this->get_parameter("stage").as_int();
    vector<cv::Point3d>points;
    for(int i = 0; i < 4; i++)
    {
        points.push_back({request->modelpoint[i].x, request->modelpoint[i].y, request->modelpoint[i].z});
        // cout << i << " " << request->modelpoint[i].x << " " << request->modelpoint[i].y << " " << request->modelpoint[i].z << endl;
        // cout << i << " " << points[i].x << " " << points[i].y << endl;
    }cout << endl;
    ShooterProcessor shooter_processor(request->g, armor_points, points, (double)request->header.stamp.nanosec * 1.0 / 1e9 + (double)request->header.stamp.sec);
    // shooter_processor.set_armor_size(points);
    cv::Point3d point = shooter_processor.get_point();
    EulerAngle angle = shooter_processor.hit_armor(point);
    response->yaw = angle.yaw;
    response->pitch = angle.pitch;
    response->roll = angle.roll;

    cout << "yaw: " << angle.yaw << "  pitch" << angle.pitch << endl;
}
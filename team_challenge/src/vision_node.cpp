#include<iostream>

#include "rclcpp/rclcpp.hpp"
#include "opencv2/opencv.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "referee_pkg/msg/multi_object.hpp"
#include "referee_pkg/msg/object.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/header.hpp"
#include "referee_pkg/msg/race_stage.hpp"
#include "digit_detector.hpp"
#include "vision_processor.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;

class VisionNode : public rclcpp::Node
{
    public:
        VisionNode(std::string name) : Node(name)
        {
            this->declare_parameter<int>("stage", 0);
            stage_ = this->get_parameter("stage").as_int();
            RCLCPP_INFO(this->get_logger(), "This is %s", name.c_str());
            image_sub = this->create_subscription<sensor_msgs::msg::Image>(
                "/camera/image_raw",
                10,
                std::bind(&VisionNode::camera_callback, this, std::placeholders::_1));

            stage_sub = this->create_subscription<referee_pkg::msg::RaceStage>(
                "/referee/race_stage",
                10,
                std::bind(&VisionNode::stage_callback, this, std::placeholders::_1));

            target_pub = this->create_publisher<referee_pkg::msg::MultiObject>(
                "/vision/target",10);

            param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&VisionNode::param_callback, this, std::placeholders::_1));

        }


    private:
        int stage_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
        rclcpp::Subscription<referee_pkg::msg::RaceStage>::SharedPtr stage_sub;
        rclcpp::Publisher<referee_pkg::msg::MultiObject>::SharedPtr target_pub;
        void camera_callback(sensor_msgs::msg::Image::SharedPtr msg);
        void stage_callback(referee_pkg::msg::RaceStage::SharedPtr msg);
        void topic_publish(vector<Target>& objects, std_msgs::msg::Header header);
        rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

        // 参数回调函数
        rcl_interfaces::msg::SetParametersResult param_callback(const std::vector<rclcpp::Parameter> &parameters) {
            auto result = rcl_interfaces::msg::SetParametersResult();
            result.successful = true;
        
            for (const auto &param : parameters) {
                if (param.get_name() == "stage") {
                    int new_stage = param.as_int();
                    if (new_stage >= 0 && new_stage <= 5) {  // stage 范围是 0-5
                        stage_ = new_stage;
                        RCLCPP_INFO(this->get_logger(), "Stage updated to: %d", stage_);
                    } else {
                        result.successful = false;
                    }
                }
            }
            return result;
        }

};

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionNode>("vision_node");
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;    
}



void VisionNode::topic_publish(vector<Target>& objects, std_msgs::msg::Header header)
{
    referee_pkg::msg::MultiObject msg_objects;
    msg_objects.header = header;
    msg_objects.num_objects = objects.size();
    for(int i = 0; i < objects.size(); i++)
    {
        referee_pkg::msg::Object obj;
        obj.target_type = objects[i].target_type;
        for(int j = 0; j < 4; j++)
        {
            geometry_msgs::msg::Point corner;
            corner.x = objects[i].points[j].x;
            corner.y = objects[i].points[j].y;
            corner.z = 0.0;
            obj.corners.push_back(corner);
        }
        msg_objects.objects.push_back(obj);
    }

    target_pub->publish(msg_objects);
    // RCLCPP_INFO(this->get_logger(), "publish %d objects:%s", msg_objects.num_objects, msg_objects.objects[0].target_type.c_str());
}

void VisionNode::camera_callback(sensor_msgs::msg::Image::SharedPtr msg){
    try{
        if(stage_ == 0 || stage_ == 5)return;
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


        vector<Target> objects;
        if(stage_ == 1)
        {
            VisionProcessor::Level1(img, objects);
        }
        else if(stage_ == 2)
        {
            VisionProcessor::Level2(img, objects);
        }
        
        topic_publish(objects, msg->header);
        

    }
    catch(cv_bridge::Exception &e){
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void VisionNode::stage_callback(referee_pkg::msg::RaceStage::SharedPtr msg)
{
    stage_ = msg->stage;
}
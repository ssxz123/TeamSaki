#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <referee_pkg/srv/hit_armor.hpp>
#include <shooter_processor.hpp>

using namespace std;
using namespace cv;

class ShooterNode : public rclcpp::Node
{
    public:
        ShooterNode(string name) : Node(name)
        {
            RCLCPP_INFO(this->get_logger(), "This is %s", name.c_str());
            hit_armor_service = this->create_service<referee_pkg::srv::HitArmor>(
                "/referee/hit_armor",
                bind(&ShooterNode::call_back, this, placeholders::_1, placeholders::_2)
            );
        }

    private:
        int stage;
        rclcpp::Service<referee_pkg::srv::HitArmor>::SharedPtr hit_armor_service;
        void call_back(
            const shared_ptr<referee_pkg::srv::HitArmor::Request> request,
            const shared_ptr<referee_pkg::srv::HitArmor::Response> response);

        
};

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    auto node = make_shared<ShooterNode>("shooter_node");
    rclcpp::spin(node);
    rclcpp::shutdown();
}

void ShooterNode::call_back(const shared_ptr<referee_pkg::srv::HitArmor::Request> request,
    const shared_ptr<referee_pkg::srv::HitArmor::Response> response)
{
    // stage = this->get_parameter("stage").as_int();
    vector<cv::Point2d>points;
    for(int i = 0; i < 4; i++)
    {
        points.push_back({request->modelpoint[i].x, request->modelpoint[i].y});
    }
    ShooterProcessor shooter_processor(request->g, points, (double)request->header.stamp.nanosec * 1.0 / 1e9 + (double)request->header.stamp.sec);

    cv::Point3d point = shooter_processor.get_point();
    EulerAngle angle = shooter_processor.hit_armor(point);
    response->yaw = angle.yaw;
    response->pitch = angle.pitch;
    response->roll = angle.roll;

    RCLCPP_INFO(this->get_logger(), "已处理服务");
}
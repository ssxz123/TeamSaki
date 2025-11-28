#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <referee_pkg/srv/hit_armor.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class ShooterNode : public rclcpp::Node
{
    public:
        ShooterNode(string name) : Node(name)
        {
            RCLCPP_INFO(this->get_logger(), "This is %s", name.c_str());
            hit_armor_client = this->create_client<referee_pkg::srv::HitArmor>("/referee/hit_armor");
        }
        void send_request(vector<Point3d>& points)
        {
            // 循环等待服务端上线：每次等待1秒，直到服务可用或节点退出
            while(!hit_armor_client->wait_for_service(std::chrono::seconds(1)))
            {
                // 检查节点是否正常运行（如用户按下Ctrl+C会导致rclcpp::ok()为false）
                if(!rclcpp::ok())
                {
                    // 若等待被打断，打印错误日志
                    RCLCPP_ERROR(this->get_logger(), "等待服务的过程中被打断...");
                    return; // 退出函数
                }
                // 服务未上线时，打印等待日志
                RCLCPP_INFO(this->get_logger(), "等待服务器上线中");
            }

            // 创建服务请求对象（智能指针），用于存储要发送的参数
            auto request = std::make_shared<referee_pkg::srv::HitArmor::Request>();
            request->modelpoint.resize(4);
            request->header.stamp = this->get_clock()->now();
            for(int i = 0; i < 4; i++)
            {
                request->modelpoint[i].x = points[i].x;
                request->modelpoint[i].y = points[i].y;
                request->modelpoint[i].z = points[i].z;
            }
            request->g = 9.8;
            

            // 异步发送服务请求：不会阻塞当前线程，服务端响应后自动调用回调函数
            hit_armor_client->async_send_request(
                request,  // 要发送的请求对象
                // 绑定结果回调函数：收到响应时执行result_callback_
                std::bind(&ShooterNode::call_back,  // 回调函数地址
                          this,  // 当前节点对象指针（绑定成员函数必须传入this）
                          std::placeholders::_1)  // 占位符：接收服务响应的future对象
            );
        }

    private:
        rclcpp::Client<referee_pkg::srv::HitArmor>::SharedPtr hit_armor_client;
        void call_back(rclcpp::Client<referee_pkg::srv::HitArmor>::SharedFuture result_future)
        {
            auto response = result_future.get();
        }

        
};

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    auto node = make_shared<ShooterNode>("test_node");

    double R = 5.0;
    Point3d C = Point3d(1.0, 2.0, 3.0);
    Point3d u2 = Point3d(1.0, 2.0, 1.0);
    Point3d u3 = Point3d(-3.0, 2.0, -1.0);
    double l2 = sqrt(u2.x * u2.x + u2.y* u2.y + u2.z * u2.z);
    double l3 = sqrt(u3.x * u3.x + u3.y* u3.y + u3.z * u3.z);
    u2.x/=l2;u2.y/=l2;u2.z/=l2;
    u3.x/=l3;u3.y/=l3;u3.z/=l3;
    for(int i = 0; i < 20; i++)
    {
        Point3d point;
        point.x = C.x + R * (cos(i * CV_PI / 20) * u2.x + sin(i * CV_PI / 20) * u3.x);
        point.y = C.y + R * (cos(i * CV_PI / 20) * u2.y + sin(i * CV_PI / 20) * u3.y);
        point.z = C.z + R * (cos(i * CV_PI / 20) * u2.z + sin(i * CV_PI / 20) * u3.z);
        vector<Point3d> points;
        points.push_back(point);
        points.push_back(point);
        points.push_back(point);
        points.push_back(point);
        node->send_request(points);
        printf("point: ( %lf, %lf, %lf )\n", point.x, point.y, point.z); 
        rclcpp::sleep_for(std::chrono::milliseconds(500));
    }





    rclcpp::spin(node);
    rclcpp::shutdown();
}

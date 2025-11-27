#ifndef SHOOTER_PROCESSOR_HPP
#define SHOOTER_PROCESSOR_HPP

#include <iostream>
#include <cstdio>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define ___ cout<<"----------"<<endl;

struct EulerAngle
{
    double yaw;
    double pitch;
    double roll;
};

class ShooterProcessor
{
    public:
        ShooterProcessor(double _g, vector<Point2d>& _points, double _t)
        {
            armorPointsTo3D(_points, obj_point);
            g = _g;
            if(t0 < 1e-6)t0 = _t;
            time = _t - t0;
            if(obj_points.size() <= 10){
                obj_points.push_back(obj_point);
                times.push_back(time);
            } 
            get_circle();
            Point3d point = get_pos_on_time(time);
            printf("time: %lf , point: ( %.5lf, %.5lf, %.5lf )\n", time, point.x, point.y, point.z);
        }

        Point3d get_point()
        {
            return obj_point;
        }
        EulerAngle hit_static_armor(Point3d point);
        EulerAngle hit_armor(Point3d point);

    private:
        double time;
        static double t0;
        double g;
        double v0 = 1.5;
        static double R;
        static Point3d C;
        static double angular;
        static double phase;
        Point3d obj_point;
        Point3d centroid;
        static Mat u2, u3;
        static vector<Point3d> obj_points;
        static vector<double> times;
        static vector<double> angles;
        bool get_circle();
        static Point3d get_pos_on_time(double t);
        // static double func(uint t);
        bool armorPointsTo3D(const vector<Point2d>& pixelPoints, Point3d& center3D);
        double func(double t);
        double dfunc(double t);
        double solvefunc();
};
double ShooterProcessor::angular;
double ShooterProcessor::phase;
double ShooterProcessor::t0 = 0.0;
double ShooterProcessor::R;
Point3d ShooterProcessor::C;
vector<Point3d> ShooterProcessor::obj_points;
vector<double> ShooterProcessor::times;
vector<double> ShooterProcessor::angles;
Mat ShooterProcessor::u2;
Mat ShooterProcessor::u3;

bool ShooterProcessor::armorPointsTo3D(const vector<Point2d>& pixelPoints, Point3d& center3D) {
    // 相机内参矩阵
    Mat cameraMatrix = (Mat_<double>(3, 3) << 
        554.383, 0, 320,
        0, 554.383, 320,
        0, 0, 1);

    // 畸变系数（假设无畸变）
    Mat distCoeffs = Mat::zeros(4, 1, CV_64F);

    // 装甲板在世界坐标系中的3D角点坐标（以装甲板中心为原点）
    // 装甲板尺寸：长0.705m，宽0.230m
    vector<Point3d> objectPoints;
    double half_length = 0.705 / 2.0;  // 长边的一半
    double half_width = 0.230 / 2.0;   // 短边的一半
    
    // 按照左下起始逆时针顺序定义角点
    // 注意：这里假设装甲板在XY平面上，Z=0
    objectPoints.push_back(Point3d(-half_length, -half_width, 0));  // 左下
    objectPoints.push_back(Point3d(half_length, -half_width, 0));   // 右下  
    objectPoints.push_back(Point3d(half_length, half_width, 0));    // 右上
    objectPoints.push_back(Point3d(-half_length, half_width, 0));   // 左上

    // 使用solvePnP求解位姿
    Mat rvec, tvec;
    bool success = solvePnP(objectPoints, pixelPoints, cameraMatrix, distCoeffs, rvec, tvec);
    
    if (!success) {
        return false;
    }

    // tvec就是相机坐标系下的装甲板中心坐标
    // 由于相机位于原点，tvec就是装甲板中心相对于相机的三维坐标
    center3D.x = tvec.at<double>(0);
    center3D.y = tvec.at<double>(1);
    center3D.z = tvec.at<double>(2);

    return true;
}

bool ShooterProcessor::get_circle()
{
    Point3d centroid_ = Point3d(0.0, 0.0, 0.0);  //求质心
    int n = obj_points.size();
    if(n < 3)return false;
    for(int i = 0; i < n; i++)
    {
        centroid_.x += obj_points[i].x;
        centroid_.y += obj_points[i].y;
        centroid_.z += obj_points[i].z;
    }
    centroid.x = centroid_.x / n;
    centroid.y = centroid_.y / n;
    centroid.z = centroid_.z / n;

    Mat data = Mat(n, 3, CV_64F);
    for(int i = 0; i < n; i++)
    {
        data.at<double>(i, 0) = obj_points[i].x - centroid.x;
        data.at<double>(i, 1) = obj_points[i].y - centroid.y;
        data.at<double>(i, 2) = obj_points[i].z - centroid.z;
    }

    PCA pca(data, Mat(), PCA::DATA_AS_ROW);     //利用PCA,获得平面向量u2,u3
    u2 = pca.eigenvectors.row(1);
    u3 = pca.eigenvectors.row(0);

    vector<Point2d>points2D;

    for(int i = 0; i < n; i++)                  //获得各点在平面上的坐标
    {
        Point3d vec = obj_points[i] - centroid;
        double x = vec.x * u2.at<double>(0) + vec.y * u2.at<double>(1) + vec.z * u2.at<double>(2);
        double y = vec.x * u3.at<double>(0) + vec.y * u3.at<double>(1) + vec.z * u3.at<double>(2);
        points2D.push_back(Point2d(x, y));
    }

    Mat A(n, 3, CV_64F);
    Mat B(n, 1, CV_64F);
    Mat X;
    for(int i = 0; i < n; i++)
    {
        double x = points2D[i].x;
        double y = points2D[i].y;
        A.at<double>(i, 0) = 2 * x;
        A.at<double>(i, 1) = 2 * y;
        A.at<double>(i, 2) = 1;
        B.at<double>(i, 0) = x * x + y * y;
    }
    solve(A, B, X, DECOMP_SVD);
    double a = X.at<double>(0,0), b = X.at<double>(0, 1), c = X.at<double>(0, 2);
    R = sqrt(a * a + b * b + c);
    C.x = centroid.x + a * u2.at<double>(0) + b * u3.at<double>(0);
    C.y = centroid.y + a * u2.at<double>(1) + b * u3.at<double>(1);
    C.z = centroid.z + a * u2.at<double>(2) + b * u3.at<double>(2);

    angles.clear();
    for(int i = 0; i < n; i++)          //  获得每个点在圆周上对应的角度
    {
        double angle = atan2(points2D[i].y - b, points2D[i].x - a);
        angles.push_back(angle);
        
    }
    for(int i = 1; i < n; i++)
    {
        double delta = angles[i] - angles[i - 1];
        if(delta > CV_PI) for(int j = i; j < n; j++) angles[j] -= 2 * CV_PI;
        else if(delta < -CV_PI) for(int j = i; j < n; j++) angles[j] += 2 * CV_PI;
    }

    Mat T(n, 2, CV_64F);
    Mat Theta(n, 1, CV_64F);
    Mat Y;
    for(int i = 0; i < n; i++)
    {
        T.at<double>(i, 0) = times[i];
        T.at<double>(i, 1) = 1.0;
        Theta.at<double>(i, 0) = angles[i];
    }
    solve(T, Theta, Y, DECOMP_SVD);
    angular = Y.at<double>(0);
    phase = Y.at<double>(1);
    

    return true;
}

Point3d ShooterProcessor::get_pos_on_time(double t)
{
    if(obj_points.size() < 3)return obj_points.back();
    double angle = angular * t + phase;

    double x = C.x + R * cos(angle) * u2.at<double>(0) + R * sin(angle) * u3.at<double>(0);
    double y = C.y + R * cos(angle) * u2.at<double>(1) + R * sin(angle) * u3.at<double>(1);
    double z = C.z + R * cos(angle) * u2.at<double>(2) + R * sin(angle) * u3.at<double>(2);

    return Point3d(x, y, z);
}

double ShooterProcessor::func(double t)
{
    double t1 = time + t;
    Point3d p = get_pos_on_time(t1);
    double r = sqrt(p.x * p.x + p.y * p.y);
    double res = pow(v0 * t0, 2) - r * r - pow(p.z * p.z + 0.5 * g * t * t, 2);
    return res;
}

double ShooterProcessor::dfunc(double t)
{
    double dt = 1e-6;
    double df = (func(t + dt) - func(t))/ dt;
    return df;
}

double ShooterProcessor::solvefunc()
{
    double t = time;
    double t_next;
    while(fabs(t_next - t) < 1e-6)
    {
        t_next = t - func(t) / dfunc(t);
    }
    return t_next;
}

EulerAngle ShooterProcessor::hit_static_armor(Point3d point)    //击打静态目标
{
    //公式：
    //  \varphi=\arctan\frac{y}{x}
    //  \theta=\arctan\left[\frac{v_0^2}{gr}\left(1\pm\sqrt{1-\frac{2gz}{v_0^2}-\frac{g^2r^2}{v_0^4}}\right)\right]

    double x = point.x, y = point.y, z = point.z;
    double yaw = 0.0, pitch = 0.0, roll = 0.0;
    yaw = atan2(y, x);
    double r = sqrt(x * x + y * y);
    if (r < 1e-6) {  // 避免除以零（1e-6 是允许的误差范围）
        // cerr << "Error: r = 0 (x and y are both zero)!" << endl;
        return {0.0, 0.0, 0.0};
    }
    double delta = 1 - 2 * g * z / (v0 * v0) - g * g * r * r / pow(v0, 4);
    if (delta < 1e-6) {
        // cerr << "Error: delta < 0 (unreachable target)! delta = " << delta << endl;
        // cerr << "Target: r = " << r << ", z = " << z << ", v0 = " << v0 << endl;
        return {0.0, 0.0, 0.0};
    }
    double tmp = v0 * v0 / g / r * (1 - sqrt(delta));
    pitch = atan(tmp);
    return {yaw, pitch, roll};
}

EulerAngle ShooterProcessor::hit_armor(Point3d point)           //击打动态目标
{
    int n = obj_points.size();
    if(n < 3)
    {
        return hit_static_armor(point);
    }
    
    double t = solvefunc();

    Point3d p = get_pos_on_time(time + t);

    return hit_static_armor(p);
}

#endif
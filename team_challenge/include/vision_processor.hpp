#ifndef VISION_PROCESSOR_HPP
#define VISION_PROCESSOR_HPP

#include "opencv2/opencv.hpp"
#include "digit_detector.hpp"
#include <iostream>

#define COLOR_RED Scalar(0, 0, 255)
#define COLOR_BLUE Scalar(255, 0, 0)
#define COLOR_GREEN Scalar(0, 255, 0)
#define COLOR_BLACK Scalar(0, 0, 0)
#define COLOR_GRAY Scalar(128, 128, 128)

using namespace cv;
using namespace std;

const double PI = 3.1415926;

// HSV颜色范围定义，每个颜色对应6个值：[H_min, H_max, S_min, S_max, V_min, V_max]
vector<vector<int>>color_hsv = {{0, 180, 0, 255, 0, 46},
                                {0, 10, 43, 255, 46, 255},
                                {156, 180, 43, 255, 46, 255},
                                {0, 180, 0, 30, 211, 255},
                                {11, 25, 43, 255, 46, 255},
                                {26, 34, 43, 255, 46, 255},
                                {35, 77, 43, 255, 46, 255},
                                {78, 99, 43, 255, 46, 255},
                                {100, 124, 43, 255, 46, 255},
                                {125, 155, 43, 255, 46, 255}
                                };

struct Armors
{
    vector<Point> points;
    int number;
};

struct Circle
{
    vector<Point> points;
    int color;
};

struct Rectangle
{
    vector<Point> points;
    int color;
};

struct Ring
{
    vector<Point> points;
};

struct Arrow
{
    vector<Point> points;
};

struct Target
{
    vector<Point> points;
    string target_type;
};

class Type
{
    public:
        static const int black = 0, red = 1, red_2 = 2, white = 3, orange = 4, yellow = 5, green = 6, cyan = 7, blue = 8, purple = 9;
        static const int sphere = 0, rectangle = 1, armor = 2;
};

class VisionProcessor
{
    public:
        // 第一级视觉处理：检测圆形和矩形目标
        static void Level1(Mat img, vector<Target>& objects);
        // 第二级视觉处理：检测装甲板目标
        static void Level2(Mat img, vector<Target>& armors);
        static void Stage1(Mat img, vector<Target>& objects);
        static void Stage2(Mat img, vector<Target>& objects);

    private:
        // 从图像中提取装甲板
        static Mat get_armors(const Mat& src, vector<Armors>& armors);
        // 从图像中提取圆形目标
        static Mat get_circles(const Mat& src, vector<Circle>& circles);
        // 从图像中提取矩形目标
        static Mat get_rects(const Mat& src, vector<Rectangle>& rects);

        static Mat get_ring(const Mat& src, vector<vector<Point>>& ring);

        static Mat get_arrow(const Mat& src, vector<Point>& arrow);

        // 根据颜色类型获取二值化掩码
        static Mat get_color(const Mat& src, const int color_type);
        // 在特定颜色通道上获取轮廓
        static Mat get_contours_on_color(Mat& src, vector<vector<Point>>& contours, const int color_type);
        // 在边缘获取轮廓
        static Mat get_contours_on_canny(Mat& src, vector<vector<Point>>& contours);
        // 形状检测：判断轮廓是圆形、矩形还是装甲板
        static string shapes_detect(const Mat& src, vector<Point>& contour, int color_type);
        // 对点集进行排序
        static void point_sort(vector<Point>& point, int type);
        //欧几里德距离
        static double d(Point a, Point b)   
        {
            return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
        }

        // 在图像上绘制装甲板
        static void draw_armors(Mat& src, vector<Armors>& armors);
        // 在图像上绘制圆形目标
        static void draw_circles(Mat& src, vector<Circle>& circles);
        // 在图像上绘制矩形目标
        static void draw_rects(Mat& src, vector<Rectangle>& rects);
        // 绘制圆环
        static void draw_ring(Mat& src, vector<vector<Point>>& ring);
        // 绘制箭头
        static void draw_arrow(Mat& src, vector<Point>& arrow);
        
        // 判断轮廓是否为圆形
        static bool is_circle(const vector<Point>& contour);
        // 判断轮廓是否为装甲板
        static bool is_armor(const Mat& src, const vector<Point>& contour);

        // 将颜色类型转换为字符串
        static string color_to_str(int color_type);

        
};

string VisionProcessor::color_to_str(int color_type)
{
    if(color_type == 0) return "black";
    if(color_type == 1) return "red";
    if(color_type == 3) return "white";
    if(color_type == 4) return "orange";
    if(color_type == 5) return "yellow";
    if(color_type == 6) return "green";
    if(color_type == 7) return "cyan";
    if(color_type == 8) return "blue";
    if(color_type == 9) return "purple";
}

void VisionProcessor::Stage1(Mat img, vector<Target>& objects)
{
    vector<vector<Point>>ring;
    get_ring(img, ring);
    draw_ring(img, ring);
    for(int i = 0; i < ring.size(); i++) objects.push_back({ring[i], "Ring_red"});

    imshow("Image Result", img);
    waitKey(1);
}

void VisionProcessor::Stage2(Mat img, vector<Target>& objects)
{
    vector<Point> arrow;
    get_arrow(img, arrow);
    draw_arrow(img, arrow);
    objects.push_back({arrow, "arrow"});
    imshow("Image Result", img);
    waitKey(1);
}

void VisionProcessor::Level1(Mat img, vector<Target>& objects)
{
    vector<Circle> circles;
    vector<Rectangle> rects;
    get_circles(img, circles);
    get_rects(img, rects);
    draw_circles(img, circles);
    draw_rects(img, rects);

    // for(int i = 0; i < circles.size(); i++) objects.push_back({circles[i].points, "circle_" + color_to_str(circles[i].color)});
    // for(int i = 0; i < rects.size(); i++) objects.push_back({rects[i].points, "rect_" + color_to_str(rects[i].color)});

    for(int i = 0; i < circles.size(); i++) objects.push_back({circles[i].points, "sphere"});
    for(int i = 0; i < rects.size(); i++) objects.push_back({rects[i].points, "rect"});

    imshow("Image Result", img);
    waitKey(1);
}

void VisionProcessor::Level2(Mat img, vector<Target>& objects)
{
    Mat img_contour;
    vector<Armors> armors;
    img_contour = get_armors(img, armors);
    draw_armors(img_contour, armors);

    for(int i = 0; i < armors.size(); i++) objects.push_back({armors[i].points, "armor_red_" + to_string(armors[i].number)});

    imshow("Image Armors", img_contour);
    waitKey(1);
}

bool VisionProcessor::is_circle(const vector<Point>& contour)
{
    // 通过周长和面积计算圆形度，判断是否为圆形
    double C = arcLength(contour, true), S = contourArea(contour);
    if ((S * 4 * PI / (C * C)) > 0.7){
        return true;
    }
    else return false;
}

bool VisionProcessor::is_armor(const Mat& src, const vector<Point>& contour)
{
    // 通过检测轮廓内红色区域数量判断是否为装甲板
    Mat img = src.clone();
    Rect roi = boundingRect(contour);
    Mat img_armor = img(roi);
    vector<vector<Point>> contours;
    get_contours_on_color(img_armor, contours, Type::red);
    if(contours.size() == 2)
    {
        return true;
    }
    else return false;
}

void VisionProcessor::draw_ring(Mat& src, vector<vector<Point>>& ring)
{
    for(int i = 0; i < ring.size(); i++)
    {
        Rect boundRect = boundingRect(ring[i]);
        for(int j = 0; j < ring[i].size(); j++)
        {
            
            circle(src, ring[i][j], 1, COLOR_GREEN, 1);
            string text = to_string(j + 1);
            putText(src, text, {ring[i][j].x, ring[i][j].y + 5}, FONT_HERSHEY_PLAIN, 0.7, COLOR_BLUE, 1);
            
        }
        string text2 = "ring_" + to_string(i + 1);
        putText(src, text2, {boundRect.x - 5, boundRect.y - 5}, FONT_HERSHEY_PLAIN, 0.7, COLOR_BLACK, 1);
    }
}

void VisionProcessor::draw_arrow(Mat& src, vector<Point>& arrow)
{
    for(int i = 0; i < arrow.size(); i++)
    {
        circle(src, arrow[i], 1, COLOR_GREEN, 1);
        string text = to_string(i + 1);
        putText(src, text, {arrow[i].x, arrow[i].y + 5}, FONT_HERSHEY_PLAIN, 0.7, COLOR_BLUE, 1);
    }
}

void VisionProcessor::draw_circles(Mat& src, vector<Circle>& circles)
{
    for(int i = 0; i < circles.size(); i++){
        Rect boundRect = boundingRect(circles[i].points);
        for(int j = 0; j < circles[i].points.size(); j++)
        {
            circle(src, circles[i].points[j], 1, COLOR_GREEN, 1);
            string text = to_string(j + 1);
            putText(src, text, {circles[i].points[j].x, circles[i].points[j].y + 5}, FONT_HERSHEY_PLAIN, 0.7, COLOR_BLUE, 1);
        }
        string text = "circle_" + color_to_str(circles[i].color);
        putText(src, text, {boundRect.x - 5, boundRect.y - 5}, FONT_HERSHEY_PLAIN, 0.7, COLOR_BLACK, 1);
    }
}

void VisionProcessor::draw_rects(Mat& src, vector<Rectangle>& rects)
{
    for(int i = 0; i < rects.size(); i++){
        Rect boundRect = boundingRect(rects[i].points);
        for(int j = 0; j < rects[i].points.size(); j++)
        {
            circle(src, rects[i].points[j], 1, COLOR_GREEN, 1);
            string text = to_string(j + 1);
            putText(src, text, {rects[i].points[j].x, rects[i].points[j].y + 5}, FONT_HERSHEY_PLAIN, 0.7, COLOR_BLUE, 1);
        }
        string text = "rect_" + color_to_str(rects[i].color);
        putText(src, text, {boundRect.x - 5, boundRect.y - 5}, FONT_HERSHEY_PLAIN, 0.7, COLOR_BLACK, 1);
    }
}

void VisionProcessor::draw_armors(Mat& src, vector<Armors>& armors)
{
    // 绘制装甲板的四个角点和数字识别结果
    for(int i = 0; i < armors.size(); i++)
    {
        for(int j = 0; j < armors[i].points.size(); j++)
        {
            Point p = armors[i].points[j];
            string text;
            text += j + 1 + '0';
            circle(src , p, 2, COLOR_GREEN);
            putText(src, text, {p.x, p.y - 3}, FONT_HERSHEY_PLAIN, 1, COLOR_GREEN, 1);
        }
        string text = "number: ";
        text += armors[i].number + '0';
        putText(src, text, {armors[i].points[3].x, armors[i].points[3].y - 10}, FONT_HERSHEY_PLAIN, 1, COLOR_RED, 1);
    }
}

void VisionProcessor::point_sort(vector<Point>& point, int type)
{
    // 对点集进行排序，type=2时为矩形排序
    if(point.size() != 4)
    {
        return;
    }
    if(type == 1)       //type : cirle
    {

    }
    else if(type == 2)  //type : rectangle
    {
        sort(point.begin(), point.end(), [](Point a, Point b){
            return a.y > b.y;
        });
        if(point[0].x > point[1].x)swap(point[0], point[1]);
        if(point[2].x < point[3].x)swap(point[2], point[3]);
    }
}

Mat VisionProcessor::get_ring(const Mat& src, vector<vector<Point>>& ring)
{
    Mat img = src.clone();
    vector<vector<Point>>contours;
    get_contours_on_canny(img, contours);
    int obj_num = contours.size();
    for(int i = 0; i < obj_num; i++)
    {
        vector<Point> points;
        Rect boundRect = boundingRect(contours[i]);
        points.push_back({boundRect.tl().x, boundRect.tl().y + boundRect.br().y >> 1});
        points.push_back({boundRect.tl().x + boundRect.br().x >> 1, boundRect.br().y});
        points.push_back({boundRect.br().x, boundRect.tl().y + boundRect.br().y >> 1});
        points.push_back({boundRect.tl().x + boundRect.br().x >> 1, boundRect.tl().y});
        if(i % 2 == 1)ring.push_back(points);
    }
    sort(ring.begin(), ring.end(), [](vector<Point>&a, vector<Point>& b){
        return a[0].x < b[0].x;
    });
    return img;
}

Mat VisionProcessor::get_arrow(const Mat& src, vector<Point>& arrow)    //箭头
{
    Mat img = src.clone();
    vector<vector<Point>>contours;
    get_contours_on_color(img, contours, Type::red);
    int obj_num = contours.size();
    for(int i = 0; i < obj_num; i++)
    {
        Point2f center;
        vector<Point>points;
        float r;
        minEnclosingCircle(contours[i], center, r);
        float peri = arcLength(contours[i], true);
        vector<Point>conPoly;
        approxPolyDP(contours[i], conPoly, 0.02 * peri, true);
        vector<Point>arrow_;
        for(int j = 0; j < conPoly.size(); j++)
        {
            arrow_.push_back(conPoly[j]);
        }
        Point c;
        c.x = center.x;
        c.y = center.y;
        sort(arrow_.begin(), arrow_.end(), [&c](Point a,Point b){
            return (a.x-c.x)*(a.x-c.x)+(a.y-c.y)*(a.y-c.y) < (b.x-c.x)*(b.x-c.x)+(b.y-c.y)*(b.y-c.y);
        });     //按点到圆心距离排序
        Point u1, u2;

        if(d(arrow_[5], arrow_[2]) + d(arrow_[5], arrow_[3]) + d(arrow_[5], arrow_[0]) + d(arrow_[5], arrow_[1])
        > d(arrow_[4], arrow_[2]) + d(arrow_[4], arrow_[3]) + d(arrow_[4], arrow_[0]) + d(arrow_[4], arrow_[1]))
        swap(arrow_[5], arrow_[4]); //找到箭头顶点

        int tmp;
        double minn = 10000.0;
        for(int j = 0; j < 4; j++)  //找到距离顶点较远的两点，即为箭头两侧的点
        {
            if(d(arrow_[j], arrow_[5]) < minn)
            {
                minn = d(arrow_[j], arrow_[5]);
                tmp = j;
            }
        }
        swap(arrow_[tmp], arrow_[3]);
        minn = 10000;
        for(int j = 0; j < 3; j++)
        {
            if(d(arrow_[j], arrow_[5]) < minn)
            {
                minn = d(arrow_[j], arrow_[5]);
                tmp = j;
            }
        }
        swap(arrow_[tmp], arrow_[2]);

        //u1, u2为基底
        u1 = Point(arrow_[5].x - center.x, arrow_[5].y - center.y);
        Point p = Point(arrow_[0].x + arrow_[1].x >> 1, arrow_[0].y + arrow_[1].y >> 1);
        u2 = Point(arrow_[0].x - p.x, arrow_[0].y - p.y);
        if(u2.cross(u1) < 0) u2 = -u2;  //保证u2在u1的左侧
        arrow.push_back(c + u1 + u2);
        arrow.push_back(c + u1 - u2);
        arrow.push_back(c + -u1 - u2);
        arrow.push_back(c + -u1 + u2);
    }
    
    return img;
}

Mat VisionProcessor::get_circles(const Mat& src, vector<Circle>& circles)
{
    vector<Circle> circles_all;
    Mat img = src.clone();
    // 遍历所有颜色类型检测圆形
    for(int color_type = 0; color_type < color_hsv.size(); color_type++)
    {
        if(color_type == Type::red_2) continue;
        vector<vector<Point>> contours;
        get_contours_on_color(img, contours, color_type);
        int obj_num = contours.size();
        for(int i = 0; i < obj_num; i++)
        {
            if(VisionProcessor::shapes_detect(src, contours[i], color_type) == "circle")
            {
                circles_all.push_back({contours[i], color_type});
            }
        }
    }
    // 将检测到的圆形转换为四个特征点
    int circles_num = circles_all.size();
    for(int i = 0; i < circles_num; i++)
    {
        Rect boundRect = boundingRect(circles_all[i].points);
        vector<Point> points;
        points.push_back({boundRect.tl().x, boundRect.tl().y + boundRect.br().y >> 1});
        points.push_back({boundRect.tl().x + boundRect.br().x >> 1, boundRect.br().y});
        points.push_back({boundRect.br().x, boundRect.tl().y + boundRect.br().y >> 1});
        points.push_back({boundRect.tl().x + boundRect.br().x >> 1, boundRect.tl().y});
        circles.push_back({points, circles_all[i].color});
    }
    return img;
}

Mat VisionProcessor::get_rects(const Mat& src, vector<Rectangle>& rects)
{
    Mat img = src.clone();
    vector<Rectangle> rects_all;
    // 遍历所有颜色类型检测矩形
    for(int color_type = 0; color_type < color_hsv.size(); color_type ++)
    {
        if(color_type == Type::red_2) continue;
        vector<vector<Point>>contours;
        get_contours_on_color(img, contours, color_type);
        int obj_num = contours.size();
        for(int i = 0; i < obj_num; i++)
        {
            if(VisionProcessor::shapes_detect(src, contours[i], color_type) == "rectangle")
            {
                rects_all.push_back({contours[i], color_type});
            }
        }
    }
    // 将检测到的矩形转换为四个角点
    int rects_num = rects_all.size();
    for(int i = 0; i < rects_num; i++)
    {
        Rect boundRect = boundingRect(rects_all[i].points);
        vector<Point> points;
        points.push_back({boundRect.tl().x, boundRect.br().y});
        points.push_back({boundRect.br().x, boundRect.br().y});
        points.push_back({boundRect.br().x, boundRect.tl().y});
        points.push_back({boundRect.tl().x, boundRect.tl().y});
        rects.push_back({points, rects_all[i].color});
    }
    return img;
}

Mat VisionProcessor::get_armors(const Mat& src, vector<Armors>& armors)
{
    Mat img = src.clone();
    
    vector<vector<Point>> contours;
    vector<vector<Point>> armors_all;
    // 在黑色通道上检测装甲板轮廓
    get_contours_on_color(img, contours, Type::black);
    for(int i = 0; i < contours.size(); i++)
    {
        if(shapes_detect(src, contours[i], Type::black) == "armor")
        {
            armors_all.push_back(contours[i]);
        }
    }
    // 对每个装甲板进行数字识别
    int armors_num = armors_all.size();
    vector<Rect>armors_rect;
    vector<Armors> armors_src;
    int armor_num[100] = {};       //装甲板上的数字对应的装甲板序号
    DigitDetector d;
    d.data_train();
    for(int i = 0; i < armors_num; i++)
    {
        armors_rect.push_back(boundingRect(armors_all[i]));

        Mat img_armor(img.size(), img.type(), COLOR_GRAY);
        img(armors_rect[i]).copyTo(img_armor(armors_rect[i]));
        vector<vector<Point>>armor_contours;
        get_contours_on_color(img_armor, armor_contours, Type::red);
        vector<Point>point_objCor;
        if(armor_contours.size() != 2)
        {
            continue;
        }
        // 提取装甲板的四个特征点
        for(int j = 0; j < armor_contours.size(); j++)
        {
            Rect rect = boundingRect(armor_contours[j]);
            Point point_upper = {rect.x + rect.width / 2, rect.y};
            Point point_lower = {rect.x + rect.width / 2, rect.y + rect.height};
            point_objCor.push_back(point_upper);
            point_objCor.push_back(point_lower);
        }
        point_sort(point_objCor, 2);
        
        // 截取数字区域进行识别
        Mat img_num = img_armor(Range(armors_rect[i].y, armors_rect[i].y + armors_rect[i].height),
                                Range(armors_rect[i].x + armors_rect[i].width / 2 - (armors_rect[i].height / 2),
                                       armors_rect[i].x + armors_rect[i].width / 2 + (armors_rect[i].height / 2)));

        int num = d.digit_detector(img_num);
        
        armors.push_back({point_objCor, num});
    }

    // 按数字大小对装甲板排序
    sort(armors.begin(), armors.end(), [](Armors a, Armors b){
        return a.number < b.number;
    });

    
    return img;
}

string VisionProcessor::shapes_detect(const Mat& src, vector<Point>& contour, int color_type)
{
    // 通过轮廓逼近和几何特征进行形状识别
    vector<Point>conPoly;
    float peri = arcLength(contour, true);
    approxPolyDP(contour, conPoly, 0.02 * peri, true);
    int objCor = conPoly.size();
    double area = contourArea(contour);
    if(area < 100)return "none";
    if(objCor == 4)
    {
        if(color_type == Type::black && is_armor(src, contour)) return "armor";
        else return "rectangle";
    }
    else if(objCor > 4)
    {
        if(is_circle(contour)) return "circle";
        else if(color_type == Type::black && is_armor(src, contour)) return "armor";
        else return "others";
    }
    else{
        return "others";
    }
}

Mat VisionProcessor::get_contours_on_color(Mat& src, vector<vector<Point>>& contours, const int color_type)
{
    // 在指定颜色通道上提取轮廓
    Mat img_contour = src.clone();
    Mat mask = get_color(src, color_type);
    vector<Vec4i> hierarchy;
    vector<vector<Point>> contours_all;
    findContours(mask, contours_all, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    for(int i = 0; i < contours_all.size(); i++)
    {
        if(contourArea(contours_all[i]) > 0)contours.push_back(contours_all[i]);
    }
    drawContours(img_contour, contours, -1, COLOR_GREEN, 1);
    return img_contour;
}

Mat VisionProcessor::get_contours_on_canny(Mat& src, vector<vector<Point>>& contours)
{
    Mat img = src.clone();
    Mat img_canny;
    Mat img_red = get_color(img, Type::red);
    Canny(img_red, img_canny, 25, 75);
    imshow("Image Canny", img_canny);
    vector<Vec4i> hierarchy;
    findContours(img_canny, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
    drawContours(img, contours, -1, COLOR_GREEN, 1);
    return img;
}

Mat VisionProcessor::get_color(const Mat& src,const int color_type){
    // 根据HSV范围获取颜色掩码
    Mat img = src.clone();
    Mat img_hsv, mask;
    cvtColor(img, img_hsv, COLOR_BGR2HSV);
    Scalar lower(color_hsv[color_type][0], color_hsv[color_type][2], color_hsv[color_type][4]);
    Scalar upper(color_hsv[color_type][1], color_hsv[color_type][3], color_hsv[color_type][5]);
    inRange(img_hsv, lower, upper, mask);
    // 红色需要处理两个区间
    if(color_type == 1)
    {
        Mat mask2;
        Scalar lower(color_hsv[color_type + 1][0], color_hsv[color_type + 1][2], color_hsv[color_type + 1][4]);
        Scalar upper(color_hsv[color_type + 1][1], color_hsv[color_type + 1][3], color_hsv[color_type + 1][5]);
        inRange(img_hsv, lower, upper, mask2);
        mask = mask | mask2;
    }
    // 形态学闭操作填充空洞
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    return mask;
}

#endif
#ifndef DIGIT_DETECTOR_HPP
#define DIGIT_DETECTOR_HPP

#include "opencv2/opencv.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;

class DigitDetector
{
    public:
        void data_train();
        int digit_detector(const Mat& src);

    private:
        Ptr<KNearest> model;
        Mat data, labels;
        int K = 5;
        
        
};

void DigitDetector::data_train()
{
    string package_dir = ament_index_cpp::get_package_share_directory("team_challenge");
    string img_dir = package_dir + "/resources/printed_digits.png";
    Mat data_src = imread(img_dir);
    Mat gray, binary;
    cvtColor(data_src, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    int a = 20, m = gray.rows / a, n = gray.cols / a;
    for(int i = 0; i < n; i++)
    {
        int col = i * a;
        for(int j = 0; j < m; j++)
        {
            int row = j * a;
            Mat tmp;
            binary(Range(row, row + a), Range(col, col + a)).copyTo(tmp);
            data.push_back(tmp.reshape(0, 1));
            labels.push_back((int)j / 5);
        }
    }

    data.convertTo(data, CV_32F);

    Ptr<TrainData> tData = TrainData::create(data, ROW_SAMPLE, labels);
    model = KNearest::create();

    model->setDefaultK(K);
    model->setIsClassifier(true);
    model->train(tData);
}

int DigitDetector::digit_detector(const Mat& src)
{
    Mat img;
    img = src.clone();
    
    cvtColor(img, img, COLOR_BGR2GRAY);
    // imshow(" ", img);
    // waitKey(1000);
    threshold(img, img, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    resize(img, img, Size(20, 20));
    img.convertTo(img, CV_32F);
    img = img.reshape(1, 1);
    // Mat response;
    // model->findNearest(img, K, response);
    // int prediction = int(response.at<float>(0,0));
    int prediction = model->predict(img);
    return prediction;
}




#endif
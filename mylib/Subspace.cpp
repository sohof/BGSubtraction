#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "../include/Subspace.hpp"

using std::cout;
using std::endl;
using std::vector;
using cv::Mat;
void compress_using_SVD(const vector<Mat> &images, const double VAR_TO_RETAIN){

    Mat bgr_channels[3];
    Mat data = images.at(0).clone();
    split(data,bgr_channels);
    for (size_t i = 0; i < data.channels(); i++)
    {
        bgr_channels[i].convertTo(bgr_channels[i],CV_64FC1);
    }

    // w is sing.vales, u left eig.vecs, vt is right.eig.vecs transposed.
    Mat w_s[3], u_s[3], vt_s[3];

    for (size_t i = 0; i < data.channels(); i++)
    {
        cv::SVD::compute(bgr_channels[i], w_s[i], u_s[i], vt_s[i]);
    } // compute svd for each channel separately

    double var_retained = 0.0;
    double accumulator= 0.0;

    // calculating sum of sing.values for all 3 channels.
    const double totolVar= cv::sum(w_s[0].col(0))(0)+sum(w_s[1].col(0))(0)+sum(w_s[2].col(0))(0);
    size_t count = 0; // used as both index into sin.val array and count of # values to keep

    while(var_retained < VAR_TO_RETAIN ){
        accumulator += w_s[0].at<double>(count,0)+ w_s[1].at<double>(count,0)+ w_s[2].at<double>(count,0);
        ++count;
        var_retained = accumulator / totolVar;
    }

    const int NR_OF_COMP_USED = count;

    vector<Mat> chls; // Prepare 3 matrices
    for (size_t i = 0; i < 3; i++)
    {
        chls.push_back(Mat(data.size(),CV_64FC1,cv::Scalar(0.0)));
    }
    // Add up to rank "count"
    for (size_t i = 0; i < count; i++)
    {
        chls[0] += w_s[0].at<double>(i,0)*u_s[0].col(i)*vt_s[0].row(i);
        chls[1] += w_s[1].at<double>(i,0)*u_s[1].col(i)*vt_s[1].row(i);
        chls[2] += w_s[2].at<double>(i,0)*u_s[2].col(i)*vt_s[2].row(i);
    }

    cv::normalize(chls[0], chls[0], 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(chls[1], chls[1], 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(chls[2], chls[2], 0, 255, cv::NORM_MINMAX, CV_8UC1);
    Mat outImg;
    cv::merge(chls,outImg);

    namedWindow("Frame", cv::WINDOW_NORMAL);
    imshow("Frame",outImg);
    int k = cv::waitKey(0);

    std::string filename{"compressed_used_"};
    filename.append(std::to_string(NR_OF_COMP_USED)).append(".png");
    imwrite("../output/svd_out/"+filename ,outImg);

    cout << "Using "<< count << " of the principal components "<< endl;
    cv::destroyAllWindows();
}

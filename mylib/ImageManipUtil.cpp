
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <iostream>
#include "../include/ImageManipUtil.hpp"

using std::vector;
using cv::Mat;
/**
 * @brief Formats images in order to apply PCA. Each image is split into
 * its separate channels and then flattened into a row vector so all the channels
 * are in the same vector. Thus each in the return Mat consists represent one
 * complete 3-channel image. The nr of rows in the Mat equals the nr of images we are working on.
 * @param images
 * @return vector<Mat>
 */
Mat formatImagesForPCA(const vector<Mat> &images)
{
  unsigned NR_IMGS = images.size(); // Nr of images we have
  unsigned NR_CHANNELS = images[0].channels();
  unsigned NR_ROWS = images[0].rows; // row size of each image
  unsigned NR_COLS = images[0].cols; // col size of each image
  unsigned NR_PIXELS = NR_ROWS*NR_COLS; // so our imgs is a point in NR_OF_PIXELS dimensional space.

  Mat dst(NR_IMGS, NR_CHANNELS*NR_PIXELS , CV_64FC1);
    for(unsigned int i = 0; i < images.size(); i++)
    {
        Mat rgbchannel[3];
        cv::split(images.at(i), rgbchannel);
        Mat image_row = rgbchannel[0].clone().reshape(0,1);
        image_row.push_back(rgbchannel[1].clone().reshape(0,1));
        image_row.push_back(rgbchannel[2].clone().reshape(0,1));
        // push_back creates a row below the existing one.
        image_row = image_row.reshape(0,1); // 3 rows reshaped to one row
        image_row.convertTo(dst.row(i),CV_64FC1);
    }
    return dst;
}


/**
 * @brief Format images for manually doing PCA byt matrix mpl and not using opencv PCA class.
 * In this our first attempt is to create column vectors of the imgs
 * @param images
 * @return vector<Mat>
 */
vector<Mat> formatImagesForManualPCA(const vector<Mat> &images)
{
  unsigned NR_IMGS = images.size(); // Nr of images we have
  unsigned NR_CHANNELS = images[0].channels();
  unsigned NR_ROWS = images[0].rows; // row size of each image
  unsigned NR_COLS = images[0].cols; // col size of each image
  unsigned NR_PIXELS = NR_ROWS*NR_COLS; // so our imgs is a point in NR_OF_PIXELS dimensional space.

    vector<Mat> dataVector;
    for (size_t i = 0; i < 3; i++)
    {
       dataVector.push_back(Mat(NR_PIXELS, NR_IMGS, CV_64FC1,cv::Scalar(0.0)));
    }
    for(size_t i = 0; i < NR_IMGS; ++i)
    {
        Mat rgbchannel[3];
        cv::split(images.at(i), rgbchannel);
        // our column vector should have NR_PIXELS rows
        rgbchannel[0].reshape(0,NR_PIXELS).convertTo(dataVector.at(0).col(i),CV_64FC1);
        rgbchannel[1].reshape(0,NR_PIXELS).convertTo(dataVector.at(1).col(i),CV_64FC1);
        rgbchannel[2].reshape(0,NR_PIXELS).convertTo(dataVector.at(2).col(i),CV_64FC1);
    }
    std::cout <<"Size of Mat inside dataVector " << dataVector.at(0).size() << std::endl;
    return dataVector;
}

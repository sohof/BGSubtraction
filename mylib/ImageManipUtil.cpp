
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <iostream>
#include "../include/ImageManipUtil.hpp"

using std::vector;
using std::string;
using namespace cv;
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

/**
 * @brief Display the first N (given as param) number of principal components.
 * @param A Mat matrix consisting of eigenVectors/Principal Components as row vectors
 * @param unsigned int nr of components to display
 * @return void
 * @return void
 */
void displayComponents(const Mat &eigVecMatrix, const int NR_COMP_TO_DISPLAY, const int NR_PIXELS, const int NR_ROWS )
{

  vector<Mat> componentsToDisplay; // for display purposes
  // need to prepare for separating the concatenated vector into its channels.
  Mat img_channels[3];

  for(int i = 0; i < NR_COMP_TO_DISPLAY; ++i){

    // Extract/copy the channels information from a given eig.vector,
    // saved as row vector of size NR_PIXELS
    eigVecMatrix.row(i).colRange(0,NR_PIXELS).copyTo(img_channels[0]);
    eigVecMatrix.row(i).colRange(NR_PIXELS,2*NR_PIXELS).copyTo(img_channels[1]);
    eigVecMatrix.row(i).colRange(2*NR_PIXELS,3*NR_PIXELS).copyTo(img_channels[2]);

    // Each channel must be reshaped to appropriate image size, and normalized to correct
    // range for display purposes.
    for (size_t i = 0; i < 3; i++)
    {
        // Size of img_channels matrix after reshape is image/patch  width x height
        img_channels[i] = img_channels[i].reshape(1,NR_ROWS);
        normalize(img_channels[i], img_channels[i], 0, 255, NORM_MINMAX, CV_8UC1);
    }
    Mat compenentAsIMG; // Preparing to show the first principal component as an image.
    merge(img_channels,3,compenentAsIMG);
    componentsToDisplay.push_back(compenentAsIMG);
  }

  for(int i =0; i< NR_COMP_TO_DISPLAY; ++i){
    string frame = "PC_Comp_"+ std::to_string(i+1);
    namedWindow(frame, WINDOW_NORMAL);
    imshow(frame ,componentsToDisplay.at(i));
  }
  int k = waitKey(0);
}

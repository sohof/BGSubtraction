
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <iostream>
#include "../include/ImageManipUtil.hpp"
#include "../include/Constants.hpp"

using std::vector; 
using std::string;
using std::cout; 
using std::endl;
using namespace cv;
using namespace myConsts;

/**
 * @brief Format an image into row vector of double values suitable for further processing. 
 * The image is split into its separate channels and then flattened into a row vector so all the channels
 * are in the same vector. We still use a Mat to represent this row vector for convenience. 
 * Thus, the returned Mat represent one complete 3-channel image in CV_64FC1. 
 * @param image as 3-channel Mat
 * @return Mat (one-dimensional)
 */
Mat formatImageIntoRowVector(const Mat &image)
{
  Mat dst(1, NR_CHANNELS*image.total(), CV_64FC1);
  Mat rgbchannel[3];

  cv::split(image, rgbchannel);
  Mat row_vec = rgbchannel[0].clone().reshape(0, 1);
  row_vec.push_back(rgbchannel[1].clone().reshape(0, 1));
  row_vec.push_back(rgbchannel[2].clone().reshape(0, 1));
  // push_back creates a row below the existing one.
  row_vec = row_vec.reshape(0, 1); // 3 rows reshaped to one row
  row_vec.convertTo(dst.row(0), CV_64FC1);
    
  return dst;
}

/**
 * @brief given a row vector containing real pixel values representing a 3-channel img. Rearrange 
 * and rescales the vector to construct the img the values represent for displaying or storing. Nr of
 * pixels per channel decides how the resizing/rearranging of pixel values takes place. 
 * @param matrix A Mat containing pixel values from 3-channels flattened into a row vector.
 * @param ROWS The number of rows the image should have/has.
 * @param COLS The number of cols the image should have/has.
 * @return Mat representing the img in CV_8UC3 format.
 */
Mat constructImageFromRow(const Mat &matrix, const int ROWS, const int COLS) {

    const int NR_PIXELS = ROWS * COLS;
    Mat chs[3]; // prepare matrices for separating the concatenated vector into its channels. 

    for (int i = 0; i < NR_CHANNELS; i++)
    {
      matrix.row(0).colRange(i*NR_PIXELS,(i+1)*NR_PIXELS).copyTo(chs[i]); 
      normalize(chs[i], chs[i], 0, 255, NORM_MINMAX, CV_8UC1); // rescale pixel values.
      chs[i] = chs[i].reshape(0, ROWS); // reshape into "ROWS" nr of rows. The nr cols become correct automatically.
    }
    // i = 0, takes (0,NR_PIXELS) -> chs[0]. i = 1, takes colRange(NR_PIXELS,2*NR_PIXELS) -> chs[1]
    // i = 2  takes (2*NR_PIXELS,3*NR_PIXELS) -> chs[2].

    Mat outImg;  // This is the reconstructured image
    merge(chs,3,outImg);
  return outImg;
}

/**
 * @brief Formats images in order to apply PCA. Each image is split into
 * its separate channels and then flattened into a row vector so all the channels
 * are in the same vector. Thus each row in the return Mat consists represent one
 * complete 3-channel image. The nr of rows in the Mat equals the nr of images we are working on.
 * @param images 
 * @return Mat 
 */
Mat formatImagesForPCA(const vector<Mat> &images)
{
  const unsigned NR_IMGS = images.size(); // Nr of images we have
  const unsigned NR_PIXELS = images.at(0).total(); //A single channel of an img is a point in NR_OF_PIXELS dimensional space.

  Mat dst(NR_IMGS, NR_CHANNELS*NR_PIXELS, CV_64FC1);

    for(decltype(images.size()) i = 0; i < images.size(); i++)
    {
        formatImageIntoRowVector(images.at(i)).copyTo(dst.row(i));
    }
    return dst;
}

void separateIntoBlocks(const vector<Mat> &images,  vector<vector<Mat>> &blocks,const int NR_H_BLOCKS, const int NR_V_BLOCKS) {

  for(decltype(images.size()) n = 0; n< images.size(); ++n) {
    // row, col in inner loop refer to nr block rows and block cols. 
    // process each img in left to right manner horizontally, and top to bottom vertically.
     for(int row = 0; row < NR_V_BLOCKS; ++row) {     // outer loop deals with block vertically. 
      
        for(int col = 0; col < NR_H_BLOCKS; ++col) {  // Inner loop goes from left 
          
          Mat imgRoi(images.at(n)(Range(row*BLOCK_SIZE,row*BLOCK_SIZE + BLOCK_SIZE),
                                  Range(col*BLOCK_SIZE,col*BLOCK_SIZE+BLOCK_SIZE)));
        // indexing into vector of vectors, which will be of length NR_OF_BLOCKS
          blocks.at(row*NR_H_BLOCKS + col).push_back(imgRoi);
      }
    }

  }

}

/**
 * @brief Format images for manually doing PCA byt matrix mpl and not using opencv PCA class.
 * In this our first attempt is to create column vectors of the imgs
 * @param images
 * @return vector<Mat>
 */
vector<Mat> formatImagesForManualPCA(const vector<Mat> &images)
{
  const unsigned NR_IMGS = images.size(); // Nr of images we have
  const unsigned NR_PIXELS = images.at(0).total(); // NR_ROWS*NR_COLS; // one channel of the img is a point in NR_OF_PIXELS dimensional space.

    vector<Mat> dataVector;
    for (int i = 0; i < NR_CHANNELS; i++)
    {
       dataVector.push_back(Mat(NR_PIXELS, NR_IMGS, CV_64FC1,cv::Scalar(0.0)));
    }
    for(size_t i = 0; i < NR_IMGS; ++i)
    {
        Mat rgbchannel[NR_CHANNELS];
        cv::split(images.at(i), rgbchannel);
        // our column vector should have NR_PIXELS rows
        rgbchannel[0].reshape(0,NR_PIXELS).convertTo(dataVector.at(0).col(i),CV_64FC1);
        rgbchannel[1].reshape(0,NR_PIXELS).convertTo(dataVector.at(1).col(i),CV_64FC1);
        rgbchannel[2].reshape(0,NR_PIXELS).convertTo(dataVector.at(2).col(i),CV_64FC1);
    }
    cout <<"Size of Mat inside dataVector " << dataVector.at(0).size() << endl;
    return dataVector;
}
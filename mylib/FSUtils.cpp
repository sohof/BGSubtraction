
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <set>
#include "../include/FSUtils.hpp"
#include "../include/Constants.hpp"
#include "../include/ImageManipUtil.hpp"
using namespace cv;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using cv::Mat;
using namespace myConsts;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
/**
 * @brief used to display an image to the screen. After display it will wait until any key is pressed.
 * @param img Mat representing the img. 
 */
void displayImage(const cv::Mat &img){

  displayImage(img,DEFAULT_WIN_NAME);

}
/**
 * @brief used to display an image to the screen. After display will wait until any key is pressed.
 * @param img Mat representing the img. 
 * @param windowName a String to be used to name window used to display the img. 
 */
void displayImage(const cv::Mat &img, const string windowName){

    namedWindow(windowName, WINDOW_NORMAL);
    imshow(windowName,img);

}
/**
 * @brief Display the first N (given as param) number of principal components.
 * @param eigVecMatrix A Mat matrix consisting of eigenVectors/Principal Components as row vectors
 * @param NR_COMPS const int nr of components to display
 * @return void
 */
void displayComponents(const Mat &eigVecMatrix, const int NR_COMPS, const int ROWS, const int COLS){
  for(int i = 0; i < NR_COMPS; ++i){
    string frame = "PC_Comp_"+ std::to_string(i+1);
    displayImage(constructImageFromRow(eigVecMatrix.row(i),ROWS,COLS),frame);
  }
  int k = waitKey(0);
}
/**
 * @brief Write the image to disk. Creates a filename based on the nr of pca components that
 * were used to reconstruct the image. Stores the image to: "../output/pca_out/"+filename
 * @param img A Mat containing the img to write to disk
 * @param NR_COMPS const int nr of components used to reconstruct the image
 * @return void
 */
void writeImage(const cv::Mat &img, const int NR_COMPS) {
    String filename{"components_used_"};
    filename.append(std::to_string(NR_COMPS)).append(".png");
    imwrite("../output/pca_out/"+filename ,img);
}
/**
 * @brief createPathsFromDirectory takes a string supplied from the command line containing the path of the 
 * directory of the images to read/use. The images/frames in the directory must be named sequentuelly 
 * if it is important to read the frames in chronological order when using readImagesFromPaths. 
 * The  Usually the set of paths contains more frames than we would
 * 
 * @param sorted_set A set of filesystem::path to store the created paths in
 * @param string A string representing the directory containing files we want to create paths from 
 * @return void
 */
void createPathsFromDirectory(setOfPaths &sorted_set, string dir)
{
    cout <<"Creating filenames from dir: " << dir << endl;
    for (const auto & entry : std::filesystem::directory_iterator(dir)){
        sorted_set.insert(entry.path());
    }
}
/**
 * @brief readImagesFromPaths reads images from a set of sorted paths containing images/frames. It is assumed
 * that frames are sorted in chronological order. Usually the set of paths contains more frames than we would
 * to use. The param NR_IMGS_TO_READ controls how many of the paths in the sorted set are used. The frames   
 * read are stored in a vector<Mat> supplied as argument.
 * @param sorted_set A set of sorted paths containing the paths of the images to read.
 * @param data  A reference to a vector of Mat to store the images.
 * @param NR_IMGS_TO_READ A const int representing the nr of images to read
 * @return void
 */
void readImagesFromPaths(const setOfPaths &sorted_set, vector<Mat> &data, const int NR_IMGS_TO_READ)
{
    int nrRead = 1;
    // the sorted_set can contain more paths to img than we want to read/use. 
    for(const auto & path : sorted_set){ 
      if(path.filename().string().front() == '.') // front return a char
        continue; // skip filenames starting with '.' character.

      Mat frame = imread(path.c_str(),COLOR_CODE);
      if(DEBUG){
      cout <<"path : " << path.c_str() << endl;
      cout <<"Dim = " << frame.size() << " Dtype = " << type2str(frame.type())<< endl;
      cout << "Matrix is continous " << frame.isContinuous() <<endl;
      }
      if(frame.empty())
      {
        std::cout << "Error: could not read the image: " << std::endl;
        exit(1);
      }

      data.push_back(frame.clone());
      if(nrRead++ >= NR_IMGS_TO_READ)
        break;
    }
    cout << "Succesfully read " << data.size() << " image(s)." <<endl;
}


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

using namespace cv;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using cv::Mat;

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
void createPathsFromDirectory(setOfPaths &sorted_set, string dir)
{
    cout <<"Creating filenames from dir: " << dir << endl;
    for (const auto & entry : std::filesystem::directory_iterator(dir)){
        sorted_set.insert(entry.path());
    }
}

void readImagesFromPaths(const setOfPaths &sorted_set, vector<Mat> &data, const int NR_IMGS_TO_READ, const int COLOR_CODE)
{
    bool DEBUG = 0;
    int nrRead = 1;
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
    //   Mat imgRoi = frame(Range(70,270),Range(30,230));
    //   Mat clippedImg;
    //   imgRoi.copyTo(clippedImg);
      data.push_back(frame.clone());
      if(nrRead++ >= NR_IMGS_TO_READ)
        break;
    }
    cout << "Succesfully read " << data.size() << " image(s)." <<endl;
}

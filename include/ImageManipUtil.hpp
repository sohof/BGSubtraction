#ifndef IMAGE_MANIP_UTILS
#define IMAGE_MANIP_UTILS

#include <opencv2/core.hpp>
#include <vector>


cv::Mat formatImagesForPCA(const std::vector<cv::Mat> &images);
std::vector<cv::Mat> formatImagesForManualPCA(const std::vector<cv::Mat> &images);
void displayComponents(const cv::Mat &eigVecMatrix, const int NR_COMP_TO_DISPLAY, const int NR_PIXELS, const int NR_ROWS);

#endif

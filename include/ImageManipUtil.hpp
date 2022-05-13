#ifndef IMAGE_MANIP_UTILS
#define IMAGE_MANIP_UTILS

#include <opencv2/core.hpp>
#include <vector>


cv::Mat formatImagesForPCA(const std::vector<cv::Mat> &images);
std::vector<cv::Mat> formatImagesForManualPCA(const std::vector<cv::Mat> &images);

#endif

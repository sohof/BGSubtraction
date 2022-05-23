#ifndef MY_PCA
#define MY_PCA
#include <opencv2/core.hpp>
#include <vector>

void manualPCA(const std::vector<cv::Mat> &images, const int NR_OF_COMP_TO_USE);
void doPCA(const std::vector<cv::Mat> &images, const double VAR_TO_RETAIN);

#endif 
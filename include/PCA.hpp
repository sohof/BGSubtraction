#ifndef MY_PCA
#define MY_PCA
#include <opencv2/core.hpp>
#include <vector>

void doPCAOnPatches(const std::vector<std::vector<cv::Mat>> &blocks, std::vector<Patch> &patches);
void manualPCA(const std::vector<cv::Mat> &images, const int NR_OF_COMP_TO_USE);
void doPCA(const std::vector<cv::Mat> &images, const double VAR_TO_RETAIN);

#endif 
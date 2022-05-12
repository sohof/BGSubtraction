
#ifndef SUBSPACE_METHODS
#define SUBSPACE_METHODS

#include <opencv2/core.hpp>
#include <vector>

void compress_using_SVD(const std::vector<cv::Mat> &images, const double VAR_TO_RETAIN);



#endif
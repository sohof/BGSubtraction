#ifndef IMAGE_MANIP_UTILS
#define IMAGE_MANIP_UTILS

#include <opencv2/core.hpp>
#include <vector>

cv::Mat constructImageFromRow(const cv::Mat &matrix, const int ROWS, const int COLS);
cv::Mat formatImageIntoRowVector(const cv::Mat &image);
cv::Mat formatImagesForPCA(const std::vector<cv::Mat> &images);
std::vector<cv::Mat> formatImagesForManualPCA(const std::vector<cv::Mat> &images);

void separateIntoBlocks(const std::vector<cv::Mat> &images,  std::vector<std::vector<cv::Mat>> &blocks,const int NR_H_BLOCKS, const int NR_V_BLOCKS);

#endif

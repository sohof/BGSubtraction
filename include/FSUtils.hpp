#ifndef FS_UTILS
#define FS_UTILS

#include <opencv2/core.hpp>
#include <filesystem>
#include <vector>
#include <set>
#include <string>
typedef std::set<std::filesystem::path> setOfPaths;

std::string type2str(int type);
void createPathsFromDirectory(setOfPaths &sorted_set, std::string dir);
void readImagesFromPaths(const setOfPaths &sorted_set, std::vector<cv::Mat> &data, const int NR_IMGS_TO_READ);

void displayImage(const cv::Mat &img);
void displayImage(const cv::Mat &img, const std::string windowName);
void displayComponents(const cv::Mat &eigVecMatrix, const int NR_COMPS, const int ROWS, const int COLS);
void writeImage(const cv::Mat &img, const int NR_COMPS);

// In commandLineParser Docs they use opencv String, I am using std::string. Seems to work just fine.
const std::string keys =
        "{help h usage ? |      | options for printing this help message   }"
        "{@path          |      | path to directory of images. Must be 1st arg supplied. (default = pca) }"
        "{@process       |      | type of process to perform. Must be 2nd arg supplied. (default = pca)}"
        "{N              |1     | nr of images to read (default = 1)     }"
        "{K              |1     | nr of principal component to use (default = 1)     }"
        "{V              |0.60  | Percentage Variance to retain (default = 0.60)    }"
        ;

#endif

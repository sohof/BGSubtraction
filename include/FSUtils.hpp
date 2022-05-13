#ifndef FS_UTILS
#define FS_UTILS

#include <opencv2/core.hpp>
#include <vector>
#include <set>
#include <string>
typedef std::set<std::filesystem::path> setOfPaths;

void createPathsFromDirectory(setOfPaths &sorted_set, std::string dir);
void readImagesFromPaths(const setOfPaths &sorted_set, std::vector<cv::Mat> &data, const int NR_IMGS_TO_READ, const int COLOR_CODE);

// In commandLineParser Docs they use opencv String, I am using std::string. Seems work just fine.
const std::string keys =
        "{help h usage ? |      | options for printing this help message   }"
        "{@path          |      | path to directory of images }"
        "{@process       |      | type of process to perform}"
        "{N              |1     | nr of images to read (default = 1)     }"
        "{K              |1     | nr of principal component to use (default = 1)     }"
        "{V              |0.90  | Percentage Variance to retain (default = 0.90)    }"
        ;

#endif

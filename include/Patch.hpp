#ifndef BGS_PATCH
#define BGS_PATCH
#include <opencv2/core.hpp>

// Seems: best way define global const from c++17 onwards. Declares the variable to be an inline variable.


class Patch 
{

private:
const int ID;
cv::PCA pca;

public:
    Patch(int id);
   // ~Patch();

    int getID();

     //constants declarations
    
    static const int PNR_ROWS; // row size of each image
    static const int PNR_COLS; // col size of each image
    static const int PNR_PIXELS; // so our imgs is a point in NR_OF_PIXELS dimensional space.
};


#endif
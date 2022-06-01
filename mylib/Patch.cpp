#include <opencv2/core.hpp>
#include <iostream>
#include "../include/Patch.hpp"
#include "../include/Constants.hpp" 
#include "../include/FSUtils.hpp" // used for displayComponents 

using namespace myConsts;
using cv::Mat;

//Constructors
Patch::Patch(int id):ID(id) {

}
Patch::Patch(int id,cv::InputArray data):
ID(id),pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, PNR_MAX_COMPONENTS) {

}
void Patch::calcPCA(cv::InputArray data){

    pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, PNR_MAX_COMPONENTS);
}

/**
 * @brief Displays all principal components as images.
 * 
 */
void Patch::displayPCAComponents() const{
     displayComponents(pca.eigenvectors, pca.eigenvectors.size().height, PNR_ROWS, PNR_COLS);
}
/**
 * @brief projects vector from the original space to the principal components subspace
 * @param vec a vector containing a flattened image 
 * @return A mat containing our point/coordinates in the principal components subspace as a row vector  
 */

cv::Mat Patch::project(const cv::Mat &vec) const {
    return pca.project(vec);

}
/**
 * @brief reconstructs the original vector from the projection. So it transforms the coord
 * vector from the principal components subspace back into the original image/pixel space. 
 * @param vec with the coords in the principal components lower dimensional subspace
 * @return A Mat containing the reconstruction of the projection in original img/pixel space.
 */

Mat Patch::backProject(const Mat &vec) const {
    return pca.backProject(vec);
}





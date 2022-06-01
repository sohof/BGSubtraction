#ifndef BGS_PATCH
#define BGS_PATCH
#include <opencv2/core.hpp>


class Patch 
{
public:
    // constructors
    Patch() = default; //recommended to provide default if other cons. are defined.
    explicit Patch(int id);
    Patch(int id, cv::InputArray data);
    
    // const member functions
    int getID() const ;
    void displayPCAComponents() const;
    cv::Mat getPrincipalComponents();
    cv::Mat backProject(const cv::Mat &vec) const ;
    cv::Mat project(const cv::Mat &vec) const;

    // member functions    
    void calcPCA(cv::InputArray data);

private:
    const int ID =0;
    cv::PCA pca;
};



// Inline functions should be def. in same header as corresp. class def.

inline
cv::Mat Patch::getPrincipalComponents(){
    return pca.eigenvectors;
 };
inline
int Patch::getID() const {
    return ID;
}

#endif
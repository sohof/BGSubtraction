#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "../include/ImageManipUtil.hpp"
#include "../include/Patch.hpp"
#include "../include/Constants.hpp"
#include "../include/FSUtils.hpp"
#include <sys/time.h>
#include <sys/resource.h>
/*
Seems visual code will flag the headers as missing even if cmake makes sures they are included during the build. So I added include path in visual code c++ Configurations. Probably because the code can be compiled from withing visual code as well, altough I am not using it. */

using namespace cv;
using namespace myConsts;
using std::cout;
using std::endl;
using std::string;
using std::vector;

/**
 * @brief doPCAOnPatches takes the blocks vector as arg, containing all frames of the video 
 * sequence but subdivded into blocks. Each inner vector "blocks.at(i)" contains all frames
 * for a given block. The doPCAOnPatches computes pca components for each such block and
 * saves the computation in structure called a patch. For each block we a corresponding patch
 * and each such patch is stored in the patches vector received as argument. 
 */

void doPCAOnPatches(const vector<vector<Mat>> &blocks, vector<Patch> &patches ){

    for(decltype(blocks.size()) i = 0; i < blocks.size(); ++i) {
             
        // Reshape and stack blocks into a rowMatrix. The 3-channels will be concatenated to long row vector.
        Mat data = formatImagesForPCA(blocks.at(i));
        //cout << "Size of data matrix " << data.size() <<endl;
        Patch p(i,data); // Just calling the constructor does all calculations. 
        patches.push_back(p);
    }

}

/**
 * @brief doPCA is an old function kept around for documentation purposes
 * on how we originally approached the problem, this is only used if the
 * program is started whith the "pca" option. Which is irrelevant when we 
 * are working with blocks/patches.
 */
void doPCA(const vector<Mat> &images, const double VAR_TO_RETAIN){

    const int NR_ROWS = images[0].rows; // row size of each image which means height
    const int NR_COLS = images[0].cols; // col size of each image which means width

    // Reshape and stack images into a rowMatrix. The 3-channels will be concatenated to long row vector.
    Mat data = formatImagesForPCA(images);
    cout << "Size of data matrix " << data.size() <<endl;
    //const int maxComp = 10;
    // send empty Mat() since we don't have precomputed means, pca does it for us.
    // Consr. accepts int max_nr of comps to use. Or a double % variance to be retained 
    PCA pca(data, cv::Mat(), PCA::DATA_AS_ROW, VAR_TO_RETAIN);
    cout << "Size of principal components matrix " << pca.eigenvectors.size() <<endl;
    
    int NR_OF_COMP_USED = pca.eigenvectors.size().height;
    cout << "Using " << NR_OF_COMP_USED << " principal components for reconstruction." <<endl;

    // display some of the principal compenents
    displayComponents(pca.eigenvectors, NR_OF_COMP_USED, NR_ROWS, NR_COLS);

    // Using the first image in our data to project onto princ.comp bases.
    Mat proj_coords = pca.project(data.row(0));

    // transforming from pca coords back to image/pixel space
    Mat reconstruction = pca.backProject(proj_coords);
    reconstruction = constructImageFromRow(reconstruction,NR_ROWS,NR_COLS);  // Rearrange/rescale to actual img

    displayImage(reconstruction,"Reconstruction");
    writeImage(reconstruction,NR_OF_COMP_USED);

    waitKey(0);
    destroyAllWindows();
}


/**
 * @brief manualPCA is an old function kept around for documentation purposes
 * on how we originally approached the problem, not used at the moment.
 */
void manualPCA(const vector<Mat> &images, const int NR_OF_COMP_TO_USE){

    const int NR_ROWS = images[0].rows; // row size of each image which means height
    const int NR_COLS = images[0].cols; // col size of each image which means width
    const int NR_PIXELS = NR_ROWS*NR_COLS; // so our imgs is a point in NR_OF_PIXELS dimensional space.


    //Step 1. Prepare imgs for further processing by splitting in channels and changing type to floating point.
    vector<Mat> dataVec = formatImagesForManualPCA(images); // imgs are saved as column vecs in separate chls

    /* Step 2.
    Calculute means along a row, so it´s mean of all imgs for same pixel location.
    The mean is subtracted from each row. It's saved because we need it for recounstruction.
    means will be colum vector of (dim of image)*nr channels;
    */
    Mat means(NR_PIXELS, NR_CHANNELS, CV_64FC1, Scalar(0.0));
    for(decltype(dataVec.size()) i =0; i<dataVec.size(); ++i){  // dataVec.size() is the nr of channels
         for(int j =0; j< dataVec.at(i).rows; ++j){
            means.at<double>(j,i) = mean(dataVec.at(i).row(j))[0]; // mean returns 4 element vector
            dataVec.at(i).row(j) = dataVec.at(i).row(j) - mean(dataVec.at(i).row(j)); //somehow subtraction works
         }
    }

    /* Step 3.
    To making computations managable we calculate A.transpose * A. Instead of A*A.transpose
    This is due to that fact that we cannot have more than rank of A nonzero eig.values anyway.
    Se we have A.t()*A*v = lambda*v. Now mpl by A on both sides => A*A.t()*A*v = lambda* A*v.
    Thus we see that A*v will in fact be an eig.vec of A*A.transpose.
    We create covariance matrices for each channel. Then for each matrix calc. its eig vecs/values. These
    eig.vecs are the right eig.vecs from V.transpose in the SVD decomposition  A = U * SIGMA * V.transpose
    */
    vector<Mat> covariances, eigenVectors, eigenValues, principalComponentMats;
    Mat eigVecs,eigVals;
    for (size_t i = 0; i < NR_CHANNELS; i++)
    {
        // also tried dividing cov matrix by n-1,but no discernible difference
        covariances.push_back(dataVec.at(i).t() * dataVec.at(i)); // cov. for every channel separately
        eigen(covariances.at(i),eigVals,eigVecs); // eig vectors will be stored as rows in eigVecs
        eigenVectors.push_back(eigVecs.clone());
        eigenValues.push_back(eigVals.clone());
        int nr_of_eig_vecs =eigVecs.size().height; // height gives us the nr of eig vectors
        // Prepare empty principalComponentMats matrices for next step
        principalComponentMats.push_back(Mat(NR_PIXELS, nr_of_eig_vecs, CV_64FC1, Scalar(0.0)));
       // cout <<  eigenVectors.at(0) <<endl;

    }

    // Step 3. Calculate the actual principal Component vectors using Av.
    int eig_vec_dim =eigVecs.row(0).size().width;  // a bit unneccesary maybe since sym.matrices nrOfeigvec=dim
    for (int n = 0; n < NR_CHANNELS; n++)
    {
        for(int i = 0; i < eigVecs.size().height; ++i)
        {
            // reshape to col.vec before mpl
            principalComponentMats.at(n).col(i)= dataVec.at(0)* eigenVectors.at(n).row(i).reshape(0,eig_vec_dim); // / eigenValues.at(n).row(i);
        }
    }
    cout << "Dimension of a principal Comp Matrix = "<< principalComponentMats.at(0).size() << endl;
    cout << "Nr of principal Comp Matrices should be 3 = "<< principalComponentMats.size() << endl;
    cout << "Using "<< NR_OF_COMP_TO_USE << " of the principal components "<< endl;

    // Step 4. To demonstrate how this all works choose first image. Project to new bases via. U.t()*x
    // So we get its coords in the new princ.component bases. This is done for each channel separately.
    vector<Mat> cordsInNewBases;
    for (size_t i = 0; i < NR_CHANNELS; i++)
    {
        Mat tmp = principalComponentMats.at(i).t()*dataVec.at(i).col(0);
        cordsInNewBases.push_back(tmp);
    }

    // Step 5. Using our calc coords in the new bases take a lin.comb of the principa.comp. vectors
    // then add back the mean.
    vector<Mat> reconstruction;
    for(size_t i = 0; i < NR_CHANNELS; i++)
    {
        Mat nrPrincipals = principalComponentMats.at(i).colRange(0,NR_OF_COMP_TO_USE);
        Mat nrCoords = cordsInNewBases.at(i).rowRange(0,NR_OF_COMP_TO_USE);
        Mat tmp = nrPrincipals*nrCoords;
        normalize(tmp, tmp, 0, 255, NORM_MINMAX, CV_64FC1); // need to normalize range before adding mean
        tmp = tmp + means.col(i); // mean was calc. based range 0-255. But type is still floating numbers.
        normalize(tmp, tmp, 0, 255, NORM_MINMAX, CV_8UC1); // re-normalize type to CV_8UC1 for displaying
        tmp = tmp.reshape(0, images[0].rows);
        reconstruction.push_back(tmp);
    }

    // Step 6. Merge channels, display and save to disk.
  
    Mat outImg;
    merge(reconstruction,outImg);

    displayImage(outImg);
    writeImage(outImg,NR_OF_COMP_TO_USE);

    waitKey(0);
    destroyAllWindows();
}



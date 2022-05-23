#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "../include/ImageManipUtil.hpp"
#include "../include/Patch.hpp"
#include "../include/Constants.hpp"
/*
Seems visual code will flag the headers as missing even if cmake makes sures they are included during the build. So I added include path in visual code c++ Configurations. Probably because the code can be compiled from withing visual code as well, altough I am not using it. */

using namespace cv;
using namespace myConsts;
using std::cout;
using std::endl;
using std::string;
using std::vector;

void doPCA(const vector<Mat> &images, const double VAR_TO_RETAIN){

    const int NR_ROWS = images[0].rows; // row size of each image which means height
    const int NR_COLS = images[0].cols; // col size of each image which means width
    const int NR_PIXELS = NR_ROWS*NR_COLS; // so our imgs is a point in NR_OF_PIXELS dimensional space.

    // Reshape and stack images into a rowMatrix. The 3-channels will be concatenated to long row vector.
    Mat data = formatImagesForPCA(images);
    cout << "Size of data matrix " << data.size() <<endl;
    const int maxComp = 5;
    // send empty Mat() since we don't have precomputed means, pca does it for us.
    PCA pca(data, cv::Mat(), PCA::DATA_AS_ROW, maxComp);
    Mat eigVecMatrix = pca.eigenvectors;
    cout << "Size of principal components matrix " << data.size() <<endl;
    
    int NR_OF_COMP_USED = pca.eigenvectors.size().height;
    cout << "Using " << NR_OF_COMP_USED << " principal components for reconstruction.";

    // display some of the principal compenents
    displayComponents(eigVecMatrix, NR_OF_COMP_USED, NR_PIXELS, NR_ROWS);

    // Using the first image in our data to project onto princ.comp bases.
    Mat point = pca.project(data.row(0));
    // transforming from pca coords back to image
    Mat reconstruction = pca.backProject(point);

    normalize(reconstruction, reconstruction, 0, 255, NORM_MINMAX, CV_8UC1);
    Mat chs[3];

    reconstruction.row(0).colRange(0,NR_PIXELS).copyTo(chs[0]);
    reconstruction.row(0).colRange(NR_PIXELS,2*NR_PIXELS).copyTo(chs[1]);
    reconstruction.row(0).colRange(2*NR_PIXELS,3*NR_PIXELS).copyTo(chs[2]);
    chs[0] = chs[0].reshape(0, NR_ROWS);
    chs[1] = chs[1].reshape(0, NR_ROWS);
    chs[2] = chs[2].reshape(0, NR_ROWS);

    Mat outImg;  // This is the reconstructured image
    merge(chs,3,outImg);

    namedWindow("Reconstruction", WINDOW_NORMAL);
    imshow("Reconstruction",outImg);

    String filename{"components_used_"};
    filename.append(std::to_string(NR_OF_COMP_USED)).append(".png");
    imwrite("../output/pca_out/"+filename ,outImg);

    int k = waitKey(0);
    destroyAllWindows();
}

void manualPCA(const vector<Mat> &images, const int NR_OF_COMP_TO_USE){

    const int NR_IMGS = images.size(); // Nr of images we have
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
    for(size_t i =0; i<dataVec.size(); ++i){  // dataVec.size() is the nr of channels
         for(size_t j =0; j< dataVec.at(i).rows; ++j){
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
    for (size_t n = 0; n < NR_CHANNELS; n++)
    {
        for(size_t i = 0; i < eigVecs.size().height; ++i)
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

    // Step 6. Merge channels display and save img to disk.
    Mat outImg;
    merge(reconstruction,outImg);
    namedWindow("Frame", WINDOW_NORMAL);
    imshow("Frame",outImg);

    String filename{"components_used_"};
    filename.append(std::to_string(NR_OF_COMP_TO_USE)).append(".png");
    imwrite("../output/pca_out/"+filename ,outImg);


    int k = waitKey(0);
    destroyAllWindows();
}



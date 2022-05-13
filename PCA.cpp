#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <set>
#include <FSUtils.hpp>
#include <Subspace.hpp>

/*
Seems visual code will flag the headers as missing even if cmake makes sures they are included during the build. So I added include path in visual code c++ Configurations. Probably because the code can be compiled from withing visual code as well, altough I am not using it. */

using namespace cv;
using std::cout;
using std::endl;
using std::string;
using std::vector;
typedef std::set<std::filesystem::path> setOfPaths;

unsigned NR_IMGS; // Nr of images we have
unsigned NR_CHANNELS;
unsigned NR_ROWS; // row size of each image
unsigned NR_COLS; // col size of each image
unsigned NR_PIXELS; // so our imgs is a point in NR_OF_PIXELS dimensional space.

/**
 * @brief Formats images in order to apply PCA. Each image is split into
 * its separate channels and then flattened into a row vector so all the channels
 * are in the same vector. Thus each in the return Mat consists represent one
 * complete 3-channel image. The nr of rows in the Mat equals the nr of images we are working on.
 * @param images
 * @return vector<Mat>
 */
Mat formatImagesForPCA(const vector<Mat> &images)
{
    Mat dst(NR_IMGS, NR_CHANNELS*NR_PIXELS , CV_64FC1);
    for(unsigned int i = 0; i < images.size(); i++)
    {
        Mat rgbchannel[3];
        split(images.at(i), rgbchannel);
        Mat image_row = rgbchannel[0].clone().reshape(0,1);
        image_row.push_back(rgbchannel[1].clone().reshape(0,1));
        image_row.push_back(rgbchannel[2].clone().reshape(0,1));
        // push_back creates a row below the existing one.
        image_row = image_row.reshape(0,1); // 3 rows reshaped to one row
        image_row.convertTo(dst.row(i),CV_64FC1);
    }
    return dst;
}

void doPCA(const vector<Mat> &images, const double VAR_TO_RETAIN){

    // Reshape and stack images into a rowMatrix
    Mat data = formatImagesForPCA(images);
    cout << "Size of data matrix " << data.size() <<endl;

    // send empty Mat() since we don't have precomputed means, pca does it for us.
    PCA pca(data, cv::Mat(), PCA::DATA_AS_ROW, VAR_TO_RETAIN);
    Mat eigVecs = pca.eigenvectors;

    Mat img_channels[3];
    const int NR_PIXELS  = images[0].rows*images[0].cols; // size of one channel
    eigVecs.row(0).colRange(0,NR_PIXELS).copyTo(img_channels[0]);
    eigVecs.row(0).colRange(NR_PIXELS,2*NR_PIXELS).copyTo(img_channels[1]);
    eigVecs.row(0).colRange(2*NR_PIXELS,3*NR_PIXELS).copyTo(img_channels[2]);

    for (size_t i = 0; i < 3; i++)
    {
        img_channels[i] = img_channels[i].reshape(1,images[0].rows);
        normalize(img_channels[i], img_channels[i], 0, 255, NORM_MINMAX, CV_8UC1);
    }

    Mat eigenFace; // Preparing to show the first principal component as an image.
    merge(img_channels,3,eigenFace);

    // Using the first image in our data to project onto princ.comp bases.
    Mat point = pca.project(data.row(0));
    Mat reconstruction = pca.backProject(point);

    int NR_OF_COMP_USED = pca.eigenvectors.size().height;
    cout << "Using " << NR_OF_COMP_USED << " principal components for reconstruction.";


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

    namedWindow("Frame", WINDOW_NORMAL);
    imshow("Frame",eigenFace);
    namedWindow("Frame2", WINDOW_NORMAL);
    imshow("Frame2",outImg);

    String filename{"components_used_"};
    filename.append(std::to_string(NR_OF_COMP_USED)).append(".png");
    imwrite("../output/pca_out/"+filename ,outImg);

    int k = waitKey(0);
    destroyAllWindows();
}

int main(int argc, char** argv)
{
    const int COLOR_CODE = 1; // 0 is grayscale, 1 is bgr 3-channel color.
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        exit(0);
    }
    // Get the path to images and other program arguments.
    const int NR_OF_IMGS = parser.get<int>("N");
    const int NR_OF_COMP_TO_USE = parser.get<int>("K");
    const double VAR_TO_RETAIN = parser.get<double>("V");
    const string directoryOfImages = parser.get<string>("@path");
    const string process = parser.get<string>("@process");

    if (directoryOfImages.empty())
    {
        cout << "No directory path supplied " <<endl;
        parser.printMessage();
        exit(1);
    }

    setOfPaths sorted_set;   // set<std::filesystem::path> to hold filenames
    std::vector<Mat> images; // vector to hold images.

    createPathsFromDirectory(sorted_set,directoryOfImages); // create set of filenames
    readImagesFromPaths(sorted_set,images,NR_OF_IMGS,COLOR_CODE); // read pics to vector

    NR_IMGS = images.size(); // Nr of images we have
    NR_CHANNELS = images[0].channels();
    NR_ROWS = images[0].rows; // row size of each image
    NR_COLS = images[0].cols; // col size of each image
    NR_PIXELS = NR_ROWS*NR_COLS; // so our imgs is a point in NR_OF_PIXELS dimensional space.

    cout << "Size of image(s) width x height = " << images.at(0).size() << ". Nr of rows  = "
         << NR_ROWS <<". Nr of cols  = " << NR_COLS <<endl;
    cout << endl;

    if(process.compare("pca") == 0){
        cout <<"Performing PCA: variance to retain "<< VAR_TO_RETAIN<<  endl;
        doPCA(images,VAR_TO_RETAIN);
    }
    else if(process.compare("svd") == 0){ // SVD ONLY WORKS WITH VAR_TO_RETAIN, does accept nr of comps.
        cout <<"Performing compression by SVD, variance to retain " << VAR_TO_RETAIN << endl;
        compress_using_SVD(images,VAR_TO_RETAIN);
    }
    else{
        cout <<"Doing Nothing " << endl;
    }

return 0;
}

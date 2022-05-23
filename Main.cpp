#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <set>
#include <FSUtils.hpp>
#include <Subspace.hpp>
#include <ImageManipUtil.hpp>
#include <Patch.hpp>
#include <Constants.hpp>
#include <PCA.hpp>
/*
Seems visual code will flag the headers as missing even if cmake makes sures they are included during the build. So I added include path in visual code c++ Configurations. Probably because the code can be compiled from withing visual code as well, altough I am not using it. */

using namespace cv;
using namespace myConsts;
using std::cout;
using std::endl;
using std::string;
using std::vector;
int main(int argc, char** argv)
{

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
    // setOfPaths is a typedef set<std::filesystem::path>  imported from FSUtils.hpp to hold filenames
    setOfPaths sorted_set;   
    vector<Mat> images; // vector to hold images.

    createPathsFromDirectory(sorted_set,directoryOfImages); // create set of filenames
    readImagesFromPaths(sorted_set,images,NR_OF_IMGS,COLOR_CODE); // read pics to vector

    const int NR_H_BLOCKS = images[0].cols / BLOCK_SIZE; // how many horizontal block is the img divided into
    const int NR_V_BLOCKS = images[0].rows / BLOCK_SIZE; // how many vertical blocks

    vector<vector<Mat>> blocks(NR_H_BLOCKS * NR_V_BLOCKS); // Create vectors to hold sequence of blocks. 

    separateIntoBlocks(images,blocks,NR_H_BLOCKS,NR_V_BLOCKS);

    cout << "Size of image(s) width x height = " << images.at(0).size() << ". Nr of rows  = "
         << images[0].rows <<". Nr of cols  = " << images[0].cols <<endl;
    
    cout << "An image will be divided into " << NR_H_BLOCKS << " horizontal and " 
         << NR_V_BLOCKS << " vertical blocks of size " <<BLOCK_SIZE <<"x"<<BLOCK_SIZE<< endl;

    if(process.compare("pca") == 0){
        cout <<"Performing PCA: variance to retain "<< VAR_TO_RETAIN<<  endl;
        doPCA(images,VAR_TO_RETAIN);
    }
    else if(process.compare("manpca") == 0){
     cout <<"Performing Manual PCA: nr of principal components to use "<< NR_OF_COMP_TO_USE <<  endl;
     manualPCA(images,NR_OF_COMP_TO_USE);
    }
    else if(process.compare("svd") == 0){ // SVD ONLY WORKS WITH VAR_TO_RETAIN, does accept nr of comps.
        cout <<"Performing compression by SVD, variance to retain " << VAR_TO_RETAIN << endl;
        compress_using_SVD(images,VAR_TO_RETAIN);
    }
    else{
        cout <<"Doing Nothing " <<endl;
    }

return 0;
}

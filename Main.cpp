#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <set>
#include <FSUtils.hpp>
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


void computeProjections(const vector<vector<Mat>> &blocks, const vector<Patch> &patches, vector<Mat> &blocksAsCoords){

    auto nrBlocks = blocks.size();
    auto nrFrames = blocks.at(0).size(); // at this moment all blocks should/must have same nrFrames

    for(decltype(nrBlocks) i = 0; i < nrBlocks; ++i) { // for each of our blocks do
        // create a mat to hold all computed projections from a given block as column vectors.
        blocksAsCoords.push_back(Mat(PNR_MAX_COMPONENTS, nrFrames, CV_64FC1,cv::Scalar(0.0))); 

        auto patch = patches.at(i); // get the patch corresponding to block i . 
      
        for(decltype(nrFrames) j = 0; j< nrFrames; ++j) {  // for each of the frames belonging to block i do

            Mat frameAsRowVec = formatImageIntoRowVector(blocks.at(i).at(j));
            Mat coords = patch.project(frameAsRowVec); //
            coords = coords.reshape(1,coords.cols); // reshape row to column vector  
            coords.col(0).copyTo(blocksAsCoords.at(i).col(j));
        }
       
    }

}
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

    if (directoryOfImages.empty() || process.empty())
    {
        std::cerr << "ERROR: No directory path or process type supplied " <<endl;
        parser.printMessage();
        exit(1);
    }

    // setOfPaths is a typedef set<std::filesystem::path>  imported from FSUtils.hpp to hold filenames
    setOfPaths sorted_set;   
    vector<Mat> images; // vector to hold images.

    createPathsFromDirectory(sorted_set,directoryOfImages); // create set of filenames
    readImagesFromPaths(sorted_set,images,NR_OF_IMGS); // read pics to vector

    const int NR_H_BLOCKS = images[0].cols / BLOCK_SIZE; // # horizontal blocks the img will be divided into
    const int NR_V_BLOCKS = images[0].rows / BLOCK_SIZE; // # vertical blocks

    vector<vector<Mat>> blocks(NR_H_BLOCKS * NR_V_BLOCKS); // Create vectors to hold sequence of blocks. 
    vector<Patch> patches; // vector to hold all patches. 
    vector<Mat> blocksAsCoords; // vector to hold projections of blocks 
    separateIntoBlocks(images,blocks,NR_H_BLOCKS,NR_V_BLOCKS);

    cout << "Size of image(s) width x height = " << images.at(0).size() << ". Nr of rows  = "
         << images[0].rows <<". Nr of cols  = " << images[0].cols <<endl;
    
    cout << "An image will be divided into " << NR_H_BLOCKS << " horizontal and " 
         << NR_V_BLOCKS << " vertical blocks of size " <<BLOCK_SIZE <<"x"<<BLOCK_SIZE<< endl;

    if(process.compare("pca") == 0){
        cout <<"Performing PCA: variance to retain "<< VAR_TO_RETAIN<<  endl;
        doPCA(images,VAR_TO_RETAIN);
        
    }
    else if(process.compare("patchpca") == 0){
     cout <<"Performing PCA on patches: nr of principal components to use "<<  PNR_MAX_COMPONENTS <<  endl;
     doPCAOnPatches(blocks,patches);
     computeProjections(blocks,patches,blocksAsCoords);
        for (auto patch : patches){
            // patch.displayPCAComponents();
        }

        for(decltype(blocksAsCoords.size()) i =0;  i<blocksAsCoords.size(); ++i){
            
            auto matrix = blocksAsCoords.at(i); // the coords are saved as column vectors
            cout << matrix.size() <<endl;
            Mat m = matrix.t(); // turn col.vecs to row.vecs since our impl.functions assume row vectors.
            for(int j = 0; j<3 ; ++j){
                
                auto coords = patches.at(i).backProject(m.row(j));
                //cout<<  m.row(j) << endl;
                displayImage(constructImageFromRow(coords,PNR_ROWS,PNR_COLS));
                int k = waitKey(0);
                if (k == 113){   // the letter q, exits program.
                    destroyAllWindows();    
                    return 0;
                }
            }
        }
    }
    else{
        
        Mat A(2, 3, CV_64F);  
        //manualPCA(images,5);
        cout << A <<endl;
        cout << A.size() <<endl;
        cout <<"Doing Nothing " <<endl;
        cout << A.reshape(1,A.total()) <<endl;
    }

return 0;
}

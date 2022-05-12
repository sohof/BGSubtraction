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
/*
Seems visual code will flag the headers as missing even if cmake makes sures they are included during the build. So I added include path in visual code c++ Configurations. Probably because the code can be compiled from within visual code as well, altough I am not using it. */

using namespace cv;
using std::cout;
using std::endl;
using std::string;
using std::vector;
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
int main()
{
    
    string DATA_PATH = "../data/video/recording.mp4"; // using relative paths
    VideoCapture cap(DATA_PATH);

    namedWindow("Frame", WINDOW_NORMAL);
    if(!cap.isOpened())
    {
        cout << "Could not read video file: "  <<endl;
        return 1;
    }
    bool display = true;
    int processed_frames_nr = 0;
    int frame_counter = 0; // counter to control nr frames coming from video.
    const int FRACTION_TO_KEEP = 3; // nr to control what fraction of incoming frames
    // to actually keep.
    const int UPPER_LIMIT = 1800; // limit nr of frames we keep from video

    double f_width =  cap.get(CAP_PROP_FRAME_WIDTH);
    double f_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "Video resolution: " << f_width << "x" << f_height << endl;
    cout << "FPS = " << cap.get(CAP_PROP_FPS) << ". FOURCC = " << cap.get(CAP_PROP_FOURCC) << endl;

    while(cap.isOpened() && display){

      Mat frame, clippedFrame;
      // Read frame by frame
      cap >> frame;
      frame_counter++; // counts all of the incoming frames
      // is also used to only keep every third frame.
      if(frame.empty() || frame_counter > 3600) // 60 sec * 30 fps = about 1800 frames. Dont want more.
        break;

      if(frame_counter < 1800 || !(frame_counter % FRACTION_TO_KEEP == 0))
        continue; // skip unless we are dealing with every 3rd frame.
      else
        processed_frames_nr++; // counts the actual nr of frames we keep.

      Mat roi = frame(Range(0,600),Range(0,900)); //  Rows 0-100, cols 0,150

      roi.copyTo(clippedFrame);
      string base;
      if(processed_frames_nr <10)
        base = "000";
      else if(processed_frames_nr<100)
        base = "00";
      else
        base = "0";

      string fileName = "../data/video_frames/frame_" + base + std::to_string(processed_frames_nr) + ".jpg";
      imshow("Frame", clippedFrame);
      imwrite(fileName, clippedFrame);

      int k = waitKey(1); // Wait for a keystroke in the window

      if(k == 's') {
        cout << "Do sth if s was pressed e.g save the img " << '\n';
        display = false;
      }
    }
    cap.release();
    cout << "Nr of frames processed: " << processed_frames_nr << endl;
    string pathToSavedFrames = "../data/video_frames/";
    //cout << "Printing file names " << endl;
    std::set<std::filesystem::path> sorted_set;
    for (const auto & entry : std::filesystem::directory_iterator(pathToSavedFrames)){
        sorted_set.insert(entry.path());
    }
    // VideoWriter framesize is supplied as: width x heigh, which is the opposite of how we read the frames.
    VideoWriter outmp4("../output/mergedframes.mp4",cv::VideoWriter::fourcc('m','p','4','v'),25, Size(900,600));

    for(const auto & path : sorted_set){

      if(path.filename().string().front() == '.') // front return a char
        continue; // skip filenames starting with '.' character.

      Mat frame = imread(path.c_str(),-1);
      //cout << "Image Dimensions = " << frame.size() << endl;
      if(frame.empty())
      {
        std::cout << "Error: could not read the image: " << std::endl;
        return 1;
      }
      outmp4.write(frame);
    }
    outmp4.release();


    destroyAllWindows(); // as soon as a key is pressed it closes (runs destructor) the window(s)

return 0;

}

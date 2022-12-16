Experimenting algorithms for background subtraction in dynamic scenes.

In this readme we add some explanation of the logic/process and thinking behind the code. So we can
more easily understand if we ever need to review it. 

# How to run the program
1. Run make from whithin build folder.
2. Then run ./Main ../data/video_frames patchpca -V=.75 -N=100
 First arg is path to file, second is the string of the func to run,
 -V variance to retain (this can be used differentyl by passing an int value and treating the variable 
 as max nr of components instead.). 
 -N is the nr of frames to use from file path source supplied.
-K is NR_OF_COMP_TO_USE to use that we can specify as command line option. 
but I have commented it out in main because I specifying this in the Constants.hpp instead.

# The Process 
A full resolution image/frame is viewed as consisting of certain number of blocks. 
How many blocks we want to dividea frame into can be decided by the constant BLOCK_SIZE. 
E.g given an 1920x1080 resolution image, and block size of 60x60 we will get 1920/60 = 32 
horizontal blocks and 18 vertical block for a total of 576 blocks per frame. Each such block
represents a specific position of the frame. Block 0 is the 60x60 block of pixels located 
in the top left corner block 1 is the 60x60 block of pixels located just to the right of 
block 0 etc. And block 575 represent the 60x60 block of pixels located in the bottom right 
corner. A stream of video will consist of a given number of frames. Each individual frame 
will thus contribute 576 blocks.

In Main.cpp we create a vector "called blocks" to store other vectors. We will need as many
of these "inner" vector as the number of blocks we have decided to divide a frame into.
The declarations is : vector<vector<Mat>> blocks(NR_H_BLOCKS * NR_V_BLOCKS);
Each inner vector stores the "cutout" / "cropped" part of the original frame for all
frames in video sequence.

Now that we have our stream of frames divided/cropped into parts/blocks where each is stored
separately we can proceed with PCA for each part/block.

The Patch class is used to do pca on all frames for a given block and store the principal 
components/eigenvectors for that block. A patch is then used for projecting/backprojecting 
from the subspace. We will of course have as many patches as we have blocks. 

After we a patch containing the eigevecs/princip components for a given block
we then a patch and its corresponding smalled sized frames and project each such
on the lower dimensional subspace so that we get coords the lower-dimensional subspace.
This means now that each frame previosly stored in Mat structure now becomes just 
column vector. So after all the frames belonging to a give block have been projected
to the subspace, they can be saved in a new Mat structure. But this structure
now represents "all the frames in the video seqs belonging to a certain block". 

So by using a "vector<Mat> blocksAsCoords" we can store data from all frames for all blocks



Set up cata data for DL model
    string filePath_train_X = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/cat_train_x.txt";
    string filePath_train_Y = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/cat_train_y.txt";  
    string filePath_test_X = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/cat_test_x.txt";
    string filePath_test_Y = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/cat_test_y.txt";

    Mat1d X_train = Mat::zeros(12288, 209, CV_64F);
    Mat1d Y_train = Mat::zeros(1, 209, CV_64F);
    Mat1d X_test = Mat::zeros(12288, 50, CV_64F);
    Mat1d Y_test = Mat::zeros(1, 50, CV_64F);

    readValuesFromFileToMat(X_train, filePath_train_X);
    readValuesFromFileToMat(Y_train, filePath_train_Y);
    readValuesFromFileToMat(X_test, filePath_test_X);
    readValuesFromFileToMat(Y_test, filePath_test_Y);

cout << "Testing L-layer network on Cat image data"<<endl;


    // Set up soccer data for course 2 week 1 Regularization
    string filePath_train_X = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/socc_train_x.txt";
    string filePath_train_Y = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/socc_train_y.txt";  
    string filePath_test_X = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/socc_test_x.txt";
    string filePath_test_Y = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/socc_test_y.txt";

    Mat1d X_train = Mat::zeros(2, 211, CV_64F);
    Mat1d Y_train = Mat::zeros(1, 211, CV_64F);
    Mat1d X_test = Mat::zeros(2, 200, CV_64F);
    Mat1d Y_test = Mat::zeros(1, 200, CV_64F);
    cout << "Testing L-layer network on 2D Soccer data"<<endl;
    
    cout << "TD matrix size: "<< X_train.size <<". TD Labels matrix size: " << Y_train.size << endl;
    cout << "Test Data matrix size: "<< X_test.size <<". Test Data Labels matrix size: " << Y_test.size << endl;

    // Set upp moon dataset for testing different optimization methods. Called "moons" 
    // because the data from each of the two classes looks a bit like a crescent-shaped moon.

    string filePath_train_X = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/moonsX.txt";
    string filePath_train_Y = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/moonsY.txt";  

    Mat1d X_train = Mat::zeros(2, 300, CV_64F);
    Mat1d Y_train = Mat::zeros(1, 300, CV_64F);

    readValuesFromFileToMat(X_train, filePath_train_X);
    readValuesFromFileToMat(Y_train, filePath_train_Y);


    // Measuring time of functions:
    #include <chrono>
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    model_train_mb(X_train, Y_train, params, NUM_ITERS,LEARNING_RATE,LAMBDA, true, BATCH_SIZE); 
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    myFile <<"Time: " << ms_double.count() << "ms. ";

    // Writing to file
    myFile.open("result.txt", std::ios::app);
    if(myFile){
        myFile<<"Iterations: "<< NUM_ITERS<< ". LearnR: "<<LEARNING_RATE << ". Lambda: " << LAMBDA <<". ";
                                         
    }
    else{
        cout << "Error file not created"<<endl;
    }
    myFile <<"Time: " << ms_double.count() << "ms. ";

    auto accuracyTrain = predictAndCalcAccuracyDeep(X_train,Y_train,params);
    myFile << "AccTrain: " << accuracyTrain << "%. " <<endl;
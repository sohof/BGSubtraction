#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <FSUtils.hpp>
#include <NN_Utils.hpp>
#include <Constants.hpp>
#include <ImageManipUtil.hpp>
#include <chrono>
#include <fstream>
using namespace myConsts;

using cv::Mat;
using cv::Mat1d;
using std::cout;
using std::endl;
using std::map;
using std::string;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
std::fstream myFile;


void model_train(const Mat1d &X, const Mat1d &Y, mat1dMap &params, const int NUM_ITERS, const double LEARNING_RATE, const double LAMBDA, const bool PRINT_COST)
{
 
    auto layers = layer_sizes(X, Y); // get vector of layers sizes
    initialize_parameters(layers, params); // init.params based on layer sizes.
    
    for (int i = 1; i < NUM_ITERS+1; i++)
    {
        
        auto outputs_forward_prop = forwardProp(X, params);
        auto AL = outputs_forward_prop.first; // Last layer actications. i.e Y_hat
        // caches is a vector<pair<tuple_3, Mat1d>>, pair of (linear_cache,activ.cache)
        auto caches = outputs_forward_prop.second; // vector of caches for use in back.prop

        double cost = compute_cost_ce_L2_reg(AL,Y,params,LAMBDA); 
        mat1dMap grads = backProp_L2_regularization(AL,Y,caches,LAMBDA);

        update_parameters(params,grads,LEARNING_RATE);
        
        if ((PRINT_COST && i % 500) == 0)
        {
            cout << "Cost after iteration " << i << ": " << cost << endl;
            if (i== NUM_ITERS)
                cout<<"Cost-Iteration " <<i<< ": "<< cost << ". "; 

        }
    }
}

void model_train_mb(const Mat1d &X, const Mat1d &Y, mat1dMap &params, const int NUM_ITERS, const double LEARNING_RATE, const double LAMBDA, const bool PRINT_COST, const int MB_SIZE)
{

   auto layers = layer_sizes(X, Y);         // get vector of layers sizes
   initialize_parameters(layers, params);   // init.params based on layer sizes.
   int t = 0;                                // initializing the counter required for Adam update
   mat1dMap v_params;
   mat1dMap s_params;
   initialize_adam(params,v_params,s_params);

   const int NUM_COMPL_MB = std::floor((double)X.cols / MB_SIZE); // compute nr of complete Mini-batches
    
    for (int iter = 1; iter < NUM_ITERS+1; iter++)
    {
        double cost = 0.0;
        for (int i = 0; i < NUM_COMPL_MB+1; i++) // Go one beyond the last comp.mini-batch to process remaining. train.exs.
        {
            cv::Range curr_range(i*MB_SIZE,(i+1)*MB_SIZE); // create a the correct range for current mini-batch

            if(i == NUM_COMPL_MB) {
                 curr_range.start = NUM_COMPL_MB * MB_SIZE;
                 curr_range.end = X.cols; // set end to nr training ex. = m.      
            }
             
            auto outputs_forward_prop = forwardProp(X.colRange(curr_range), params);
            auto AL = outputs_forward_prop.first; // Last layer actications. i.e Y_hat
             // caches is a vector<pair<tuple_3, Mat1d>>, pair of (linear_cache,activ.cache)
            auto caches = outputs_forward_prop.second; // vector of caches for use in back.prop

            cost = compute_cost_ce_L2_reg(AL,Y.colRange(curr_range),params,LAMBDA);
            mat1dMap grads = backProp_L2_regularization(AL,Y.colRange(curr_range),caches,LAMBDA);

            //update_parameters(params,grads,LEARNING_RATE);
            t=t+1;
            update_parameters_adam(params,grads,v_params,s_params,t,0.9,0.999,LEARNING_RATE);

        }
        if ((PRINT_COST && iter % 500) == 0)
        {
            cout << "Cost after iteration " << iter << ": " << cost << endl;
            if (iter == NUM_ITERS)
                 myFile<<"Cost-Iteration " <<iter<< ": "<< cost << ". "; 

        }    
    }
}

int main()
{
    // Set upp moon dataset. 
    string filePath_train_X = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/moonsX.txt";
    string filePath_train_Y = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/moonsY.txt";  
    Mat1d X_train = Mat::zeros(2, 300, CV_64F);
    Mat1d Y_train = Mat::zeros(1, 300, CV_64F);
    readValuesFromFileToMat(X_train, filePath_train_X);
    readValuesFromFileToMat(Y_train, filePath_train_Y);
 
    //cout << std::setprecision(17);

    const int NUM_ITERS = 5000;
    const int BATCH_SIZE = 32;  // Batch-size canot be complete TD-set.
    const double LEARNING_RATE = 0.0007;
    const double LAMBDA = 0.0;
    mat1dMap params;

    
    cout << "TD matrix size: "<< X_train.size <<". TD Labels matrix size: " << Y_train.size << endl;
    cout <<"Iterations: " << NUM_ITERS<< ". LearnR: " <<  LEARNING_RATE 
         << ". Lambda: " << LAMBDA <<". Batch Size "<< BATCH_SIZE <<". " ;
    
    auto t1 = high_resolution_clock::now();
    model_train_mb(X_train, Y_train, params, NUM_ITERS,LEARNING_RATE,LAMBDA, true, BATCH_SIZE); 
    //model_train(X_train, Y_train, params, NUM_ITERS,LEARNING_RATE,LAMBDA, true); 
    auto t2 = high_resolution_clock::now();


    duration<double, std::milli> ms_double = t2 - t1; // Getting number of milliseconds as a double
    cout <<"Time: " << ms_double.count() << "ms. ";

    auto accuracyTrain = predictAndCalcAccuracyDeep(X_train,Y_train,params);
    cout << "AccTrain: " << accuracyTrain << "%. " <<endl;

    //auto accuracyTest = predictAndCalcAccuracyDeep(X_test,Y_test,params);
    //myFile << "AccTest: " << accuracyTest << "%." << endl;

    // myFile.close();

}

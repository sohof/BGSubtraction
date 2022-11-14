
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <FSUtils.hpp>
#include <NN_Utils.hpp>
#include <Constants.hpp>
#include <ImageManipUtil.hpp>
using namespace myConsts;

using cv::Mat;
using cv::Mat1d;
using std::cout;
using std::endl;
using std::map;
using std::string;
using std::vector;

void printMap(const mat1dMap &params){
      // iterate using C++17 facilities
    for (const auto& [key, value] : params)
        std::cout << '[' << key << "] = "<<endl << value << endl;
}
void printMatSlice(const Mat1d& matrix, int rowsToPrint, int colsToPrint)
{
    for (int i = 0; i < rowsToPrint; i++)
    {
        cout<<"Row "<<i << ": "; 
        for (int j = 0; j < colsToPrint; j++)
        {
            cout<<matrix.at<double>(i,j)<< "  ";
        }
        cout << endl;
    }
}
void nn_model_train(const Mat1d &X, const Mat1d &Y, mat1dMap &params, const int NUM_ITERS, const double LEARNING_RATE, const bool PRINT_COST)
{
    cout<<"nn_model_train. Layers (2,4,1). Nr of iterations: " << NUM_ITERS<< ". Learning rate: " <<  LEARNING_RATE << endl;
    //auto layers = layer_sizes(X, Y);
    vector<int> layers = {2,4,1};

    initialize_parameters(layers, params);
    //cout << "W2 = " << params.at("W2") << endl;
    for (int i = 0; i < NUM_ITERS; i++)
    {
        
        mat1dMap cache = forward_propagation(X, params);

        double cost = compute_cost(cache.at("A2"),Y);

        mat1dMap grads = back_prop(params, cache, X, Y);

        update_parameters(params, grads, LEARNING_RATE);
    
        if (PRINT_COST && i % 1000 == 0)
        {
            cout << "Cost after iteration " << i << ": " << cost << endl;
        }
        
    }
}

void L_layer_model(const Mat1d &X, const Mat1d &Y, mat1dMap &params, const int NUM_ITERS, const double LEARNING_RATE, const bool PRINT_COST)
{
    cout<<"L_layer_model. Nr of iterations: " << NUM_ITERS<< ". Learning rate: " <<  LEARNING_RATE << endl;

    auto layers = layer_sizes(X, Y); // get vector of layers sizes

    initialize_parameters(layers, params); // init.params based on layer sizes.

    for (int i = 0; i < NUM_ITERS; i++)
    {
        
        auto outputs_forward_prop = L_model_forward(X, params);
        auto AL = outputs_forward_prop.first; // Last layer actications. i.e Y_hat
        // caches is a vector<pair<tuple_3, Mat1d>>, pair of (linear_cache,activ.cache)
        auto caches = outputs_forward_prop.second; // vector of caches for use in back.prop

        double cost = compute_cost_deep(AL,Y);

        mat1dMap grads = L_model_backward(AL,Y,caches);
        update_parametersDeep(params,grads,LEARNING_RATE);

        if ((PRINT_COST && i % 1000) == 0)
        {
            cout << "Cost after iteration " << i << ": " << cost << endl;
        }
        
    }
}

int main()
{
    string filePathInput = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/Input.txt";
    string filePathLabel = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/Labels.txt";
    Mat1d X = Mat::zeros(2, 400, CV_64F);
    Mat1d Y = Mat::zeros(1, 400, CV_64F);
    readValuesFromFileToMat(X, filePathInput);
    readValuesFromFileToMat(Y, filePathLabel);

    string filePath_train_X = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/train_x.txt";
    string filePath_train_Y = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/train_y.txt";  
    string filePath_test_X = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/test_x.txt";
    string filePath_test_Y = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/test_y.txt";

    Mat1d X_train = Mat::zeros(12288, 209, CV_64F);
    Mat1d Y_train = Mat::zeros(1, 209, CV_64F);
    Mat1d X_test = Mat::zeros(12288, 50, CV_64F);
    Mat1d Y_test = Mat::zeros(1, 50, CV_64F);

    readValuesFromFileToMat(X_train, filePath_train_X);
    readValuesFromFileToMat(Y_train, filePath_train_Y);
    readValuesFromFileToMat(X_test, filePath_test_X);
    readValuesFromFileToMat(Y_test, filePath_test_Y);


    cout << "Testing 2-layer network on planar 2-dimensional data "<<endl;
    mat1dMap paramsSmall;
    nn_model_train(X, Y, paramsSmall, 10000, 1.2, true);
  
    Mat1d predictions = predict(X,paramsSmall);
    calcAndPrintAccuracy(predictions,Y);

    cout <<"Test planar data with deeper network" <<endl;

    mat1dMap paramsSmall_v2;
    L_layer_model(X, Y, paramsSmall_v2, 10000,1.2,true);
    Mat1d predictions_v2 = predict(X,paramsSmall_v2);
    calcAndPrintAccuracy(predictions_v2,Y);

    // cout << "Testing L-layer network on Cat image data"<<end;
    // cout << "TD matrix size: "<< X_train.size <<". TD Labels matrix size: " << Y_train.size << endl;
    // cout << "Test Data matrix size: "<< X_test.size <<". Test Data Labels matrix size: " << Y_test.size << endl;
 

    // mat1dMap params;
    // L_layer_model(X_train, Y_train, params, NR_ITERATIONS,true);

    // predictAndCalcAccuracyDeep(X_train,Y_train,params);
    // predictAndCalcAccuracyDeep(X_test,Y_test,params);




   

}

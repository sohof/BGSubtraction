
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <FSUtils.hpp>
#include <NN_Utils.hpp>
#include <Constants.hpp>

using namespace myConsts;

using cv::Mat;
using cv::Mat1d;
using std::cout;
using std::endl;
using std::map;
using std::string;
using std::vector;


void printMatSlice(const Mat1d& matrix, int rowsToPrint, int colsToPrint)
{
    for (int i = 0; i < rowsToPrint; i++)
    {
        for (int j = 0; j < colsToPrint; j++)
        {
            cout<<matrix.at<double>(i,j)<< "  ";
        }
        cout << endl;
    }
}
void nn_model_train(const Mat1d &X, const Mat1d &Y, mat1dMap &params, const int num_iters, bool print_cost)
{

    auto layers = layer_sizes(X, Y);

    initialize_parameters(layers, params);
    //cout << "W2 = " << params.at("W2") << endl;
    for (int i = 0; i < num_iters; i++)
    {
        
        mat1dMap cache = forward_propagation(X, params);
        //cout << "Print slice from A2" <<endl;
        //printMatSlice(cache.at("A2"),1,10);

        double cost = compute_cost(cache.at("A2"),Y);

        mat1dMap grads = back_prop(params, cache, X, Y);
        //cout << "dW2 = " << grads.at("dW2") << endl;

        update_parameters(params, grads, 0.9);
        //cout << "W2 = " << params.at("W2") << endl;
    
        if (print_cost && i % 1000 == 0)
        {
            cout << "Cost after iteration " << i << ": " << cost << endl;
        }
        
    }
}

int main()
{
    string filePathInput = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/Input.txt";
    string filePathLabel = "/Users/sohof/Dropbox/Code/BGSubtraction/data/TextFiles/Labels.txt";

    // Matrices for testing initial. and forward prop
    std::vector<double> xd = {1.62434536, -0.61175641, -0.52817175,
                              -1.07296862, 0.86540763, -2.3015387};

    std::vector<double> w1d = {-0.00416758, -0.00056267, -0.02136196, 0.01640271,
                               -0.01793436, -0.0084174, 0.00502881, -0.01245288};

    std::vector<double> w2d = {-0.01057952, -0.00909008, 0.00551454, 0.02292208};

    // these biases seem to be only for forward.prop.. Seems backprop uses diff.test data
    std::vector<double> b1d = {1.74481176, -0.7612069, 0.3190391, -0.24937038};
    std::vector<double> b2d = {-1.3};

    std::vector<double> b1dbprop = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> b2dbrop = {0.0};

    Mat1d t_X(2, 3, xd.data());
    Mat1d W1(4, 2, w1d.data());
    Mat1d W2(1, 4, w2d.data());
    // Mat1d b1(4, 1, b1d.data());
    // Mat1d b2(1, 1, b2d.data());

    Mat1d b1(4, 1, b1dbprop.data());
    Mat1d b2(1, 1, b2dbrop.data());

    // cout << t_X << endl;

    // these are special test case data for testing compute_cost
    // std::vector<double> yd = {1.0, 0.0, 1.0};
    // Mat1d t_Y(1, 3, yd.data());
    // cout << t_Y << endl;

    // std::vector<double> A2dat = {0.5002307, 0.49985831, 0.50023963};
    // Mat1d A2_test(1, 3, A2dat.data());

    // std::vector<double> y1 = {.9, 0.2, 0.1, .4, .9};
    // std::vector<double> y2 = {1, 0, 0, 1, 1};
    // Mat1d Y1(5, 1, y1.data());
    // Mat1d Y2(5, 1, y2.data());

    // cout << "Doing sigmoid \n" << sigmoid(C) << endl;
    // cout << "Doing sigmoid_deriv \n"      << sigmoid_derivative(C) << endl;
    // cout << "C After sigmoid  calls \n"   << C << endl;

    // cout << "Doing relu \n"  << relu(A) << endl;
    // cout << "After relu A \n"  << A << endl;
    // cout << "Doing relu_deriv \n"  << relu_derivative(A) << endl;
    // cout << "After relu A \n"  << A << endl;

    // cout << "L2 loss A \n" << l2_loss(Y1, Y2) << endl;

    Mat1d X = Mat::zeros(2, 400, CV_64F);
    Mat1d Y = Mat::zeros(1, 400, CV_64F);
    readValuesFromFileToMat(X, filePathInput);
    readValuesFromFileToMat(Y, filePathLabel);

    // cout <<"Size X = "<< X.size << " " <<" Size Y = " <<Y.size() << endl;

    // auto layers = layer_sizes(X, Y);
    //  std::map<string, Mat1d> params;
    //  std::map<string, Mat1d> cache;
    //  std::map<string, Mat1d> grads;
    //  mat1dMap params;
    //  mat1dMap cache;
    //  mat1dMap grads;
    //  initialize_parameters(layers,params);

    // params.emplace("W1", W1);
    // params.emplace("W2", W2);
    // params.emplace("b1", b1);
    // params.emplace("b2", b2);

    // cout << params["W1"] << endl;
    // cout << params["W2"] << endl;
    // cout << params["b1"] << endl;
    // cout << params["b2"] << endl;

    // forward_propagation(t_X, params, cache);

    // cout << "Z1 = " << cache["Z1"] << endl;
    // cout << "A1 = " << cache["A1"] << endl;
    // cout << "Z2 = " << cache["Z2"] << endl;
    // cout << "A2 = " << cache["A2"] << endl;

    // compute_cost(cache["A2"], t_Y);

    // back_prop(params, cache, t_X, t_Y, grads);

    // cout << "dW1 = " << grads["dW1"] << endl;
    // cout << "db1 = " << grads["db1"] << endl;

    // cout << "dW2 = " << grads["dW2"] << endl;
    // cout << "db2 = " << grads["db2"] << endl;

    // update_parameters(params, grads, 1.2);
    // cout << params["W1"] << endl;
    // cout << params["W2"] << endl;
    // cout << params["b1"] << endl;
    // cout << params["b2"] << endl;

    //cout << std::setprecision(17)<<endl;

    mat1dMap params;
    nn_model_train(X, Y, params, 30000, true);
  
    Mat1d predictions = predict(X,params);

    calcAndPrintAccuracy(predictions,Y);
}

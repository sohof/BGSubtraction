#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "../include/NN_Utils.hpp"
#include "../include/Constants.hpp"

using cv::Mat;
using cv::Mat1d;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::map; 

Mat1d sigmoid(const Mat1d& data)
{

    Mat1d s = data.clone();
    exp(-s, s);
    s = 1 / (1 + s);

    return s;
}

Mat1d sigmoid_derivative(const Mat1d& data)
{

    Mat1d tmp = sigmoid(data);
    Mat1d ds;
    cv::multiply(tmp, 1 - tmp, ds);
    return ds;
}

Mat1d tanh(const Mat1d& data)
{

    Mat1d negExp = -1*data.clone();
    Mat1d posExp = data.clone();
    exp(negExp, negExp);
    exp(posExp, posExp);
     
    Mat res =  (posExp - negExp) / (posExp + negExp) ;

    return res;
}
Mat1d tanh_derivative(const Mat1d& data)
{
    Mat1d tmp;//  = data.clone();
    pow(data,2.0,tmp);
    Mat1d res = (1-tmp);

    return res;
}

Mat1d relu(const Mat1d &data)
{

    Mat1d tmp = data.clone();
    return max(0, tmp);
}

Mat1d leaky_relu(const Mat1d &data)
{

    Mat1d tmp = data.clone();
    return max(0.01*tmp, tmp);
}
Mat1d relu_derivative(const Mat1d &data)
{

    Mat1d res = Mat::zeros(data.size(), data.type());

    for (int i = 0; i < data.rows; i++)
    {
        for (int j = 0; j < data.cols; j++)
        {
            if (data.at<double>(i, j) <= 0)
            {
                res.at<double>(i, j) = 0.0;
            }
            else
            {
                res.at<double>(i, j) = 1.0;
            }
        }
    }

    return res;
}
Mat1d leaky_relu_derivative(const Mat1d &data)
{

    Mat1d res = Mat::zeros(data.size(), data.type());

    for (int i = 0; i < data.rows; i++)
    {
        for (int j = 0; j < data.cols; j++)
        {
            if (data.at<double>(i, j) <= 0)
            {
                res.at<double>(i, j) = 0.01;
            }
            else
            {
                res.at<double>(i, j) = 1.0;
            }
        }
    }

    return res;
}
double l2_loss(const Mat1d& yhat, const Mat1d& y)
{

    return (y - yhat).dot(y - yhat);
}

vector<int> layer_sizes(const Mat1d& X, const Mat1d& Y)
{

    vector<int> lsizes;
    lsizes.push_back(X.rows);

    for (auto i : myConsts::hidden_layers)
    {

        lsizes.push_back(i);
    }

    lsizes.push_back(Y.rows);

    return lsizes;
}

void initialize_parameters(const vector<int> &layer_dims, std::map<string, Mat1d> &params)
{

    const int L = layer_dims.size();
    const double low = -1.0;
    const double high = 1.0;
 
    const string W = "W";
    const string b = "b";
    for (int l = 1; l < L; l++)
    {
        string matrix_name = W + std::to_string(l);
        string bias_param_name = b + std::to_string(l);

        Mat1d w_matrix(layer_dims.at(l), layer_dims.at(l-1), CV_64F);
        cv::randu(w_matrix, cv::Scalar(low), cv::Scalar(high));
        w_matrix = w_matrix * 0.01;
        Mat1d bias_vector = Mat::zeros(layer_dims.at(l),1, w_matrix.type());

        params.emplace(matrix_name,w_matrix);
        params.emplace(bias_param_name,bias_vector);
    }
}


mat1dMap forward_propagation(const Mat1d &X, const mat1dMap& params)
{
    mat1dMap cache; 
    Mat1d Z1 = params.at("W1")*X  + cv::repeat(params.at("b1"),1,X.cols);
    cache.emplace("Z1",Z1);
    
    Mat1d A1 = tanh(Z1);  
    cache.emplace("A1",A1);   
   
    Mat1d Z2 = params.at("W2")*A1 + cv::repeat(params.at("b2"),1,X.cols);; 
    cache.emplace("Z2",Z2);

    Mat1d A2 = sigmoid(Z2);  
    cache.emplace("A2",A2);

    return cache;
}

// OBS Might need to re-implement the sum(logProbs) to use reduce instad.
double compute_cost(const Mat1d& Y_hat, const Mat1d& Y)
{
    const int m = Y.cols;
    double cost = 0.0;
    
    Mat1d logY_hat; 
    Mat1d logOneMinusY_hat; 
    log(Y_hat,logY_hat);
    log((1-Y_hat),logOneMinusY_hat);

    Mat1d logProbs = Y.mul(logY_hat) + (1-Y).mul(logOneMinusY_hat);
    // sum calcs per channel up to 4 channels. Use [0] to get summed elms from first channel.
    cost = (-1.0/m)*cv::sum(logProbs)[0]; // We should only have 1-channel, since this is not an img.
    return cost;
}


mat1dMap back_prop(const mat1dMap& params, const mat1dMap& cache,const Mat1d& X,const Mat1d& Y)
{
    const int m = X.cols;
    mat1dMap grads;  

    Mat1d dZ2 = cache.at("A2") - Y;

    Mat1d dW2 = (1.0/m)*(dZ2*cache.at("A1").t()) ;
    grads.emplace("dW2",dW2);

    Mat1d db2;
    cv::reduce(dZ2,db2,1,cv::REDUCE_SUM,CV_64F); // sum along each row to get a row sum
    db2 = (1.0/m)*db2; // divide
    grads.emplace("db2",db2);


    //Mat1d act_deriv = tanh_derivative(cache.at("A1"));
    Mat1d dZ1 = (params.at("W2").t()*dZ2).mul(tanh_derivative(cache.at("A1")));
   
    Mat1d dW1 = (1.0/m)*(dZ1*X.t());
    grads.emplace("dW1",dW1);

    Mat1d db1;
    cv::reduce(dZ1,db1,1,cv::REDUCE_SUM,CV_64F); // sum along each row to get a row sum
    db1 = (1.0/m)*db1; // divide
    grads.emplace("db1",db1);

    return grads;

}
mat1dMap& update_parameters(mat1dMap& params, mat1dMap& grads, double learning_rate)
{
    Mat1d W1 = params.at("W1");// .clone();
    Mat1d b1 = params.at("b1");//.clone();
    Mat1d W2 = params.at("W2");//.clone();
    Mat1d b2 = params.at("b2");//.clone();

    W1 = W1 - learning_rate*grads.at("dW1");
    b1 = b1 - learning_rate*grads.at("db1");
    W2 = W2 - learning_rate*grads.at("dW2");
    b2 = b2 - learning_rate*grads.at("db2");

    params.clear();
    params.emplace("W1",W1);
    params.emplace("b1",b1);
    params.emplace("W2",W2);
    params.emplace("b2",b2);
    
    return params;
}
Mat1d predict(const cv::Mat1d& X, const mat1dMap& params)
{
    mat1dMap cache = forward_propagation(X,params);

    Mat1d predictions = (cache.at("A2") > 0.5) / 255.0;

   return predictions; 
}

void calcAndPrintAccuracy(const cv::Mat1d &predictions, const cv::Mat1d &Y)
{
    // t ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
    Mat1d accuracyMat = ((Y*predictions.t()) + (1-Y)*(1-predictions.t())) / (double) Y.cols ;

    cout <<"Accuracy: " << accuracyMat.at<double>(0,0) * 100 << "%" <<endl;
}
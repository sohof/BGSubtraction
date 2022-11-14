#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <cmath>
#include "../include/NN_Utils.hpp"
#include "../include/Constants.hpp"

using cv::Mat;
using cv::Mat1d;
using cv::reduce;
using std::cout;
using std::endl;
using std::map;
using std::pair;
using std::string;
using std::tuple;
using std::vector;

// **** FUNCTIONS USED IN BOTH TYPE OF NETWORKS ****

// l2_loss not used at all right now.
double l2_loss(const Mat1d &yhat, const Mat1d &y)
{

    return (y - yhat).dot(y - yhat);
}

// Returns vector of ints representing size of input, size of hiddden layers and size of output
vector<int> layer_sizes(const Mat1d &X, const Mat1d &Y)
{

    vector<int> lsizes;
    lsizes.push_back(X.rows);
    cout << "Layer sizes: " << X.rows;
    for (auto i : myConsts::hidden_layers)
    {
        lsizes.push_back(i);
        cout <<" " << i ;     
    }
    cout <<" " << Y.rows << endl;
    lsizes.push_back(Y.rows);

    return lsizes;
}

/**
 * @brief Initialize our W parameter matrices using (Not sure if I have the name correct)
 * Xavier initialization. Where matrix size is init. using standard normal gaussian followed
 * by multiplication with sqrt(1 / n_(l-1)). Where is n_(l-1) = nr of input/hidden units
 * from previous layer. This means, assuming we are at layer l, we look at how many
 * inputs are coming to current layer l, initialize weights of this layer proportionally
 * to the number of inputs that are coming in. This scheme works well for sigmoid activations.
 * If using ReLU using sqrt(2 / n_(l-1)) is supposed to make network train better. 
 * @param layer_dims 
 * @param params 
 */
void initialize_parameters(const vector<int> &layer_dims, std::map<string, Mat1d> &params)
{

    const int L = layer_dims.size();
    const double mean = 0.0;
    const double stddev = 1.0;

    const string W = "W";
    const string b = "b";
    for (int l = 1; l < L; l++)
    {
        string matrix_name = W + std::to_string(l);
        string bias_param_name = b + std::to_string(l);

        Mat1d w_matrix(layer_dims.at(l), layer_dims.at(l - 1), CV_64F);
        // Not sure about the name, but think it is Xavier initialization
        // normal stand.distr. of right size * sqrt(1 / nr of hidden units from prev.layer)
        cv::randn(w_matrix, cv::Scalar(mean), cv::Scalar(stddev));
        w_matrix = w_matrix * std::sqrt(2.0/layer_dims.at(l - 1));
        Mat1d bias_vector = Mat::zeros(layer_dims.at(l), 1, w_matrix.type());

        params.emplace(matrix_name, w_matrix);
        params.emplace(bias_param_name, bias_vector);
    }
}

// ***** START OF FUNCTIONS FOR DEEP L Layer NETWORK ********

/**
 * @brief Implements the sigmoid activation.
 * In this our first attempt is to create column vectors of the imgs
 * @param Z -- Mat1d of any shape

 * @return A pair<Mat1d,Mat1D> consisting of (A,cache)
          A -- output of sigmoid(Z), same shape as Z
          cache -- which is equal to Z, useful during backpropagation
 */
pair<Mat1d, Mat1d> sigmoid_deep(const Mat1d &Z)
{

    Mat1d A = Z.clone(); // Cloning might be unecessary
    exp(-A, A);
    A = 1.0 / (1 + A);
    assert(A.size == Z.size);
    Mat1d cache = Z;   // probably unecessary
    return {A, cache}; // list initalizing the pair.
}

/**
 * @brief Implement the backward propagation for a single SIGMOID unit.
 * @param dA -- post-activation gradient, of any shape
 * @param cache  -- 'Z' where we store for computing backward propagation efficiently
 * @return dZ   -- Gradient of the cost with respect to Z
 */
Mat1d sigmoid_backward(const Mat1d &dA, const Mat1d &cache)
{

    Mat1d Z = cache;
    Mat1d s;
    exp(-Z, s);
    s = 1.0 / (1 + s);

    Mat1d sigDeriv = s.mul(1 - s); // impl. elem.wise mpl dZ = dA * s * (1-s)
    Mat1d dZ = dA.mul(sigDeriv);

    assert(dZ.size == Z.size);
    return dZ;
}

/**
 * @brief Implements the RELU activation function.
 * @param Z -- Output of the linear layer, of any shape

 * @return A pair<Mat1d,Mat1D> consisting of (A,cache)
          A -- Post-activation parameter, of the same shape as Z.
          cache -- which is equal to Z, stored for computing the backward pass efficiently.
 */
pair<Mat1d, Mat1d> relu_deep(const Mat1d &Z)
{

    Mat1d A = max(0, Z.clone()); // Cloning might be unecessary

    assert(A.size == Z.size);
    Mat1d cache = Z;   // probably unecessary
    return {A, cache}; // list initalizing the pair.
}

/**
 * @brief Implement the backward propagation for a single RELU unit.
 * @param dA -- post-activation gradient, of any shape
 * @param cache -- 'Z' where we store for computing backward propagation efficiently
 * @return dZ -- Gradient of the cost with respect to Z
 */
Mat1d relu_backward(const Mat1d &dA, const Mat1d &cache)
{

    Mat1d Z = cache;
    Mat1d dZ = dA.clone();

    // our dA/dZ is the post-activation gradient propagating backwards. E.g in the last layer
    // it would be dL/dA. i.e derivative of cost funct w.r.t activation function at last layer, abbrev. dA.
    // when prop.gating backwards it supposed to be mpl with dA/dZ, which is deriv.actic.funct w.r.t input z
    // z. So we are performing: dL/dZ = (dL/dA)*(dA/DZ), and we are returning (dL/dZ). abbreviated as dZ.
    // We have previusly saved Z values in our cache convenient for calc dA/DZ. 
    // Remember: Z = W*(A:inputfromprevlayer)+b. 
    // Now: since: dA/dZ for the relu function = 0 anywhere <=0 and 1 elsewhere. And this matriz is supposed
    // to mpl with the matrix dA. So we just copy dA over and set it to 0 wherever Z= 0, and leave other 
    // values as they since they would be mpl 1.0 anyways. 
    for (int i = 0; i < Z.rows; i++)
    {
        for (int j = 0; j < Z.cols; j++)
        {
            if (Z.at<double>(i, j) <= 0)
            {
                dZ.at<double>(i, j) = 0.0;
            } // I am not setting dZ.at<double>(i, j) = 1.0; otherwize
        }
    }

    return dZ;
}

pair<Mat1d, Mat1d> tanh_deep(const Mat1d &Z)
{
    Mat1d cache = Z.clone();
    Mat1d negExp = -1 * Z.clone(); // Cloning might be unecessary
    Mat1d posExp = Z.clone();      // Cloning might be unecessary
    exp(negExp, negExp);
    exp(posExp, posExp);

    Mat A = (posExp - negExp) / (posExp + negExp);

    return {A,cache};
}
Mat1d tanh_backward(const Mat1d &dA, const Mat1d &cache)
{

    Mat1d tmp; //  = data.clone(); // deriv. is 1-tahn(z)^2
    Mat1d tangensH = tanh(cache);
    pow(tangensH, 2.0, tmp);
    Mat1d res = (1 - tmp);

    Mat1d dZ = dA.mul(res);
    return dZ;
}

pair<Mat1d, tuple_3> linear_forward(const Mat1d &A, const Mat1d &W, const Mat1d &b)
{
    //cout<<"inside linear_forward"<< endl;
    //cout<<"W size =  "<< W.size << " A size " << A.size << endl;
    // A.cols arg: nr to repeat b along horiz. axis. 1=repeat along.vertical axis.
    Mat1d Z = (W * A) + cv::repeat(b, 1, A.cols);
    auto cache = std::make_tuple(A, W, b);

    return {Z, cache};
}

/**
 * @brief Implement the forward propagation for the LINEAR->ACTIVATION layer
 * 
 * @param A_prev activations from previous layer (or input data): (size of previous layer, number of examples)
 * @param W weights matrix: Mat1d of shape (size of current layer, size of previous layer)
 * @param b bias vector, Mat1d of shape (size of the current layer, 1)
 * @param activation the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
 * @return pair<Mat1d, pair<tuple_3, Mat1d>> (A,cache)
 * A -- Matd1d the output of the activation function, also called the post-activation value 
 * cache -- A pair: containing "linear_cache" as 3-tuple and "activation_cache" as Mat1d. Values are 
 * stored for computing the backward pass efficiently
 */
pair<Mat1d, pair<tuple_3, Mat1d>> linear_activation_forward(const Mat1d &A_prev, const Mat1d &W, const Mat1d &b, string activation)
{

    tuple_3 linear_cache;
    pair<tuple_3, Mat1d> cache;
    Mat1d A;
    if (activation == "sigmoid")
    {
        //cout<<"inside linear_activation_forward sigmoid" << endl;

        auto lin_fram_par = linear_forward(A_prev, W, b); // returns (Z,(A,W,b))
        linear_cache = lin_fram_par.second;               // = 3-tuple (A,W,b)
        auto act_fram_par = sigmoid_deep(lin_fram_par.first); // send in Z. Get (A,activation_cache=Z)
        A = act_fram_par.first;
        cache = std::make_pair(linear_cache, act_fram_par.second); // (linear_cache,activation_cache)
    }
    else // If not sigmoid, then must have been "relu"
    {
        //cout<<"inside linear_activation_forward relu" << endl;
        auto lin_fram_par = linear_forward(A_prev, W, b); // returns (Z,(A,W,b))
        linear_cache = lin_fram_par.second;               // = 3-tuple (A,W,b)
        auto act_fram_par = relu_deep(lin_fram_par.first); // send in Z. Get (A,activation_cache=Z)   *** CHANGED RELUD TO TANH
        A = act_fram_par.first;
        cache = std::make_pair(linear_cache, act_fram_par.second); // (linear_cache,activation_cache)
    }

    return {A, cache};
}

pair<Mat1d,vector<pair<tuple_3, Mat1d>>> L_model_forward(const Mat1d &X, const mat1dMap &params){
   
   //cout<<"inside L_model_forward"<< endl;

    vector<pair<tuple_3, Mat1d>> caches;
    const int L = params.size() / 2; // divide by 2 to get nr Layers in network 
    const string W = "W";
    const string b = "b"; 
    Mat1d A = X;
    Mat1d A_prev;
    // First we do [LINEAR -> RELU]*(L-1). Save the "cache" to the "caches" list.
    // The for loop starts at 1 because layer 0 is the input
    for (int l = 1; l < L; ++l)
    {
        string W_name = W + std::to_string(l);
        string b_name = b + std::to_string(l);

        A_prev = A;
        auto par = linear_activation_forward(A_prev,params.at(W_name),params.at(b_name),"relu");
        A = par.first; // par.first is "A" (a Mat1d) returned from linear_activation_forward
        // par.second is the cache from linear_activation_forward. 
        caches.push_back(par.second); // par.second=cache=(linear_cache,activation_cache) 
    }
    // Last forward step LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    string W_name = W + std::to_string(L);
    string b_name = b + std::to_string(L);
    auto par = linear_activation_forward(A,params.at(W_name),params.at(b_name),"sigmoid");
    Mat1d AL = par.first;  // Activations of Last layer. i.e Y-hat
    caches.push_back(par.second); // save (linear_cache,activation_cache) from last layer.
    return {AL,caches}; // return a pair  
}


double compute_cost_deep(const Mat1d &AL, const Mat1d &Y)
{
    const double m = Y.cols;
    double cost = 0.0;

    Mat1d log_AL;
    Mat1d oneMinusAL = 1-AL;
    Mat1d logOneMinusAL;
    log(AL, log_AL);
    log(oneMinusAL, logOneMinusAL);

    Mat1d res = -(Y*log_AL.t()) -((1-Y)*logOneMinusAL.t());
    cost = (1.0/m) *cv::sum(res)[0];
    return cost;
}
   
/**
 * @brief Implement the linear portion of backward propagation for a single layer (layer l)
 * 
 * @param dZ -- Gradient of the cost with respect to the linear output (of current layer l)
 * @param linear_cache tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
 * @return tuple_3 (dA_prev, dW,db) where,
 *  dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
 *  dW -- Gradient of the cost with respect to W (current layer l), same shape as W
 *  db -- Gradient of the cost with respect to b (current layer l), same shape as b
 */
tuple_3 linear_backward(const Mat1d &dZ, tuple_3 linear_cache)
{
    Mat1d A_prev = std::get<0>(linear_cache);
    Mat1d W = std::get<1>(linear_cache);
    Mat1d b = std::get<2>(linear_cache);
    const int m = A_prev.cols;
    
    Mat1d dW = (1.0 / m) * (dZ*A_prev.t());

    Mat1d db;   
    reduce(dZ,db,1,cv::REDUCE_SUM,CV_64F);
    db = (1.0 / m) * db; 

    Mat1d dA_prev = W.t()*dZ;

    return std::make_tuple(dA_prev,dW,db);
}
/**
 * @brief  Implement the backward propagation for the LINEAR->ACTIVATION layer. 
 *
 * @param dA -- post-activation gradient for current layer l 
 * @param cache  -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
 * @param activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
 * @return tuple_3 (dA_prev,dW,db) where,
 *  dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
 *  dW -- Gradient of the cost with respect to W (current layer l), same shape as W
 *  db -- Gradient of the cost with respect to b (current layer l), same shape as b
 */
tuple_3 linear_activation_backward(const Mat1d &dA, const pair<tuple_3, Mat1d> &cache, string activation)
{
    tuple_3 linear_cache = cache.first;
    Mat1d activation_cache = cache.second;
    
    Mat1d dA_prev;
    if (activation == "relu")
    {
        //cout<<"inside linear_activation_backward relu" << endl;
        Mat1d dZ = relu_backward(dA,activation_cache); 
        return linear_backward(dZ,linear_cache); // returns (dA_prev,dW,db) as 3-tuple
    }
    else // if it was not "relu" then must be sigmoid.
    {
        Mat1d dZ = sigmoid_backward(dA,activation_cache); 
        return linear_backward(dZ,linear_cache); // returns (dA_prev,dW,db) as 3-tuple
    }

}

/**
 * @brief Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
 * 
 * @param AL -- probability vector, output of the forward propagation (L_model_forward())
 * @param Y -- true "label" vector (containing 0 if absence of target class, 1 if presence of target class)
 * @param caches -- list of caches containing:
 * every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
 * the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
 * @return mat1dMap -- A map (string,Mat1d) with the gradients
 *            grads.at("dA" + str(l)) = ... 
 *            grads.at("dW" + str(l)) = ...
 *            grads.at("db" + str(l)) = ... 
 */
mat1dMap L_model_backward(const Mat1d &AL, const Mat1d &Y, const vector<pair<tuple_3, Mat1d>> &caches)
{
    mat1dMap grads;

    const int L = caches.size(); // nr of Layers in network 
    //const int m = AL.cols; // AL is our matrix of predictions/y_hat (dim_of_input x nr of examples)
    const string dA = "dA"; 
    const string dW = "dW"; 
    const string db = "db"; 

    //cout <<"L = " << L << " m = "<< m << endl;
    Mat1d dAL = -((Y/AL) -( (1.0 - Y)/(1.0 - AL) )); 
    //cout << dAL << endl;

    pair<tuple_3,Mat1d> current_cache = caches.at(L-1);
    tuple_3 tp3 = linear_activation_backward(dAL,current_cache,"sigmoid");

    Mat1d dA_prev_tmp = std::get<0>(tp3);
    Mat1d dW_tmp = std::get<1>(tp3);
    Mat1d db_tmp = std::get<2>(tp3);

    grads.emplace((dA+std::to_string(L-1)),dA_prev_tmp);
    grads.emplace((dW+std::to_string(L)),dW_tmp);
    grads.emplace((db+std::to_string(L)),db_tmp);

    for (int l=L-2; l >= 0; --l) // we need to loop backwards from l=L-2 to l=0
    {
        pair<tuple_3,Mat1d> current_cache = caches.at(l);
        tuple_3 tp3 = linear_activation_backward(grads.at((dA+std::to_string(l+1))),current_cache,"relu");

        Mat1d dA_prev_tmp = std::get<0>(tp3);
        Mat1d dW_tmp = std::get<1>(tp3);
        Mat1d db_tmp = std::get<2>(tp3);

        grads.emplace((dA+std::to_string(l)),dA_prev_tmp);
        grads.emplace((dW+std::to_string(l+1)),dW_tmp);
        grads.emplace((db+std::to_string(l+1)),db_tmp);

    }
    return grads;
}
void update_parametersDeep(mat1dMap &params, const mat1dMap &grads, const double LEARNING_RATE)
{
    const int L = params.size() / 2; // divide by 2 to get nr Layers in network 
 
    for (int l = 0; l < L; l++)
    {
        const string W_key = "W" + std::to_string(l+1);
        const string b_key = "b" + std::to_string(l+1);
        const string dW_key = "dW" + std::to_string(l+1);
        const string db_key = "db" + std::to_string(l+1);

        Mat1d W = params.at(W_key).clone(); 
        Mat1d b = params.at(b_key).clone();

         W = W - (LEARNING_RATE * grads.at(dW_key));
         b = b - (LEARNING_RATE * grads.at(db_key)); 
    
        // params.insert_or_assign(W_key,W);
        // params.insert_or_assign(b_key,b);

        params.erase(W_key);
        params.erase(b_key);
        params.emplace(W_key,W);
        params.emplace(b_key,b);
    }
    
}

void predictAndCalcAccuracyDeep(const cv::Mat1d &X, const cv::Mat1d &Y, const mat1dMap &params)
{

    const int m = X.cols;

    Mat1d p = Mat::zeros(1, m, CV_64F);
    int numCorrect = 0;
    auto output =  L_model_forward(X, params); // returns a pair (mat1d,vector of caches)
    Mat1d AL = output.first; //  We only need the final output
    const int row = 0; // there should be only one row in AL for classif.case 
    for (int j = 0; j < m; j++)
    {
        if(AL.at<double>(row,j) > 0.5)
        {
            p.at<double>(row,j) = 1.0; 
        }
        else
        {
            p.at<double>(row,j) = 0.0;
        }

        if (p.at<double>(row,j) == Y.at<double>(row,j))
        {
            numCorrect++;
        }
    }

    double accuracy = numCorrect / (double) m;
    cout << "Accuracy: " << accuracy * 100 << "%" << endl;

}
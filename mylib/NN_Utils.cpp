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
using std::to_string;

// **** PRINT FUNCTIONS USED Debugging to Print Maps, Matrix slices  ****

void printMap(const mat1dMap &params){
      // iterate using C++17 facilities
   //int layer_to_skip = 1;
    for (const auto& [key, value] : params){
        //if(layer_to_skip++ <= 2)
          //  continue;
        std::cout << '[' << key << "] = " << value.size() << endl;
        cout << value << endl;
    }
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

// **** FUNCTIONS USED MISC INITIALIZATIONS   ****

/**
 * @brief Returns vector of ints representing size of input, size of hiddden layers and size of output
 * 
 * @param X input data matrix
 * @param Y output ground truth matrix
 * @return vector<int> 
 */
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

    for (int l = 1; l < L; l++)
    {
        string matrix_name = Wstr + std::to_string(l);
        string bias_param_name = bstr + std::to_string(l);

        Mat1d w_matrix(layer_dims.at(l), layer_dims.at(l - 1), CV_64F);
        // Not sure about the name, but think it is Xavier initialization
        // normal stand.distr. of right size * sqrt(1 / nr of hidden units from prev.layer)
        cv::randn(w_matrix, cv::Scalar(mean), cv::Scalar(stddev));
        w_matrix = w_matrix * std::sqrt(1.0/layer_dims.at(l - 1));
        Mat1d bias_vector = Mat::zeros(layer_dims.at(l), 1, w_matrix.type());

        params.emplace(matrix_name, w_matrix);
        params.emplace(bias_param_name, bias_vector);
    }
}

/**
 * @brief Initialize the Adam variables v_params and s_params, which are maps of (string,Mat1d)
 * and keep track of the past information. Their keys are the same as for grads, that is:
 * keys: "dW1", "db1", ..., "dWL", "dbL" 
 * values: Mat1d matrices of zeros of the same shape as the corresponding gradients/parameters.
 * @param params map of our params used to get correct shape of matrices
 * @param v_params that will contain the exponentially weighted average of the gradient. Initialized with zeros.
 * @param s_params that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
 */
void initialize_adam(const mat1dMap &params, mat1dMap &v_params, mat1dMap &s_params)
{
    const int L = params.size() / 2; // divide by 2 to get nr Layers in network 

    for (int l = 0; l < L; l++) // 
    {
        const string layer_str = std::to_string(l+1);
        const string matrix_name = dW + layer_str;
        const string bias_param_name = db + layer_str;

        Mat1d VdW_matrix = Mat::zeros(params.at(Wstr+layer_str).size(),CV_64F);
        Mat1d Vdb_bias_vec = Mat::zeros(params.at(bstr+layer_str).size(), CV_64F);

        v_params.emplace(matrix_name, VdW_matrix);
        v_params.emplace(bias_param_name, Vdb_bias_vec);

        // Must create separate matrices below, otherwise maps become interlinked!!  
        Mat1d SdW_matrix = Mat::zeros(params.at(Wstr+layer_str).size(),CV_64F);
        Mat1d Sdb_bias_vec = Mat::zeros(params.at(bstr+layer_str).size(), CV_64F);

        s_params.emplace(matrix_name, SdW_matrix);
        s_params.emplace(bias_param_name, Sdb_bias_vec);
    }

}


// ***** ACTIVATION FUNCTIONS   ********

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

    Mat1d A = Z.clone(); // Cloning DOES SEEM TO be Necessary! Alg.diverges otherwise.
    exp(-A, A);
    A = 1.0 / (1 + A);

    assert(A.size == Z.size);
    return {A, Z}; // list initalizing the pair. Doing Mat1d cache = Z; was unecessary
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

    Mat1d A = max(0, Z); // Did not need to clone 
    assert(A.size == Z.size);
    return {A, Z}; // list initalizing the pair. To do Mat1d cache = Z;  was unecessary
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
    Mat1d dZ = dA;  // Did not need to clone

    // our dA/dZ is the post-activation gradient propagating backwards. E.g in the last layer
    // it would be dL/dA. i.e derivative of cost funct w.r.t activation function at last layer, abbrev. dA.
    // when prop.gating backwards it supposed to be mpl with dA/dZ, which is deriv.actic.funct w.r.t input z
    // z. So we are performing: dL/dZ = (dL/dA)*(dA/DZ), and we are returning (dL/dZ). abbreviated as dZ.
    // We have previusly saved Z values in our cache convenient for calc dA/DZ. 
    // Remember: Z = W*(A:inputfromprevlayer)+b. 
    // Now: since: dA/dZ for the relu function = 0 anywhere <=0 and 1 elsewhere. And this matrix is going
    // to mpl with the matrix dA. We simplify by just copying dA over and set it to 0 wherever Z= 0, and leave other 
    // values as they are since they would be mpl 1.0 anyways. 
    for (int i = 0; i < Z.rows; i++)
    {
        for (int j = 0; j < Z.cols; j++)
        {
            if (Z.at<double>(i, j) <= 0)
            {
                dZ.at<double>(i, j) = 0.0;
            }
        }
    }

    return dZ;
}

pair<Mat1d, Mat1d> tanh_deep(const Mat1d &Z)
{
    Mat1d cache = Z.clone();
    Mat1d negExp = -1 * Z; // Cloning might be unecessary
    Mat1d posExp = Z;      // Cloning might be unecessary
    exp(negExp, negExp);
    exp(posExp, posExp);

    Mat A = (posExp - negExp) / (posExp + negExp);

    return {A,cache};
}
Mat1d tanh_backward(const Mat1d &dA, const Mat1d &cache)
{

    Mat1d tmp; //  = data.clone(); // deriv. is 1-tahn(z)^2
    Mat1d tangensH = tanh_deep(cache).first; // tanh_deep rets a pair {A,cache}, we want the "A"
    cv::pow(tangensH, 2.0, tmp);
    Mat1d res = (1 - tmp);

    Mat1d dZ = dA.mul(res);
    return dZ;
}


// **** COST FUNCTIONS USED ****
/**
 * @brief Computes to MSE of two vectors/matrices.

 * @param AL Matrix of predictions/final output
 * @param Y Matrix of ground truth
 * @return double 
 */
double compute_cost_mse(const Mat1d &AL, const Mat1d &Y)
{
    Mat1d diffsquared;
    cv::pow(AL-Y,2,diffsquared);
    double cost = cv::sum(diffsquared)[0]/ (AL.cols*AL.rows);
    return  cost;
}

/**
 * @brief Compute the cross-entropy cost
 * @param AL probability vector corresponding to your label predictions, shape (1, number of examples)
 * @param Y true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
 * @return double cross-entropy cost
 */
double compute_cost_ce(const Mat1d &AL, const Mat1d &Y)
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
 * @brief Compute the cross-entropy cost with L2-regularization. 
 * @param AL probability vector corresponding to your label predictions, shape (1, number of examples)
 * @param Y true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
 * @return double cross-entropy cost
 */
double compute_cost_ce_L2_reg(const Mat1d &AL, const Mat1d &Y, const mat1dMap &params, double lambd)
{
    const double m = Y.cols; // Nr of training samples
    const int L = params.size() / 2; // divide by 2 to get nr Layers in network 
    double cross_entropy_cost = compute_cost_ce(AL,Y);
    double regularization_cost = 0;
    
    for (int l = 0; l < L; ++l)
    {
        string W_key = Wstr + to_string(l+1);

        Mat1d elm_wize_squared;
        cv::pow(params.at(W_key),2.0,elm_wize_squared);

        regularization_cost += cv::sum(elm_wize_squared)[0];

    }
    regularization_cost *= (lambd)/(2.0*m);

    return (cross_entropy_cost + regularization_cost);
}


// **** FORWARD PROPAGATION FUNCTIONS  ****

/**
 * @brief Implement the linear part of a layer's forward propagation.
 * @param A activations from previous layer (or input data): (size of previous layer, number of examples)
 * @param W weights matrix: of shape (size of current layer, size of previous layer)
 * @param b bias vector, of shape (size of the current layer, 1)
 * @return pair<Mat1d, tuple_3> = (Z, cache) where:
 * Z - the input of the activation function, also called pre-activation parameter 
 * cache - a 3-tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
 */
pair<Mat1d, tuple_3> linear_forward(const Mat1d &A, const Mat1d &W, const Mat1d &b)
{
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
        auto lin_fram_par = linear_forward(A_prev, W, b); // returns (Z,(A,W,b))
        linear_cache = lin_fram_par.second;               // = 3-tuple (A,W,b)
        auto act_fram_par = sigmoid_deep(lin_fram_par.first); // send in Z. Get (A,activation_cache=Z)
        A = act_fram_par.first;
        cache = std::make_pair(linear_cache, act_fram_par.second); // (linear_cache,activation_cache)
    }
    else // If not sigmoid, then must have been "relu"
    {
        auto lin_fram_par = linear_forward(A_prev, W, b); // returns (Z,(A,W,b))
        linear_cache = lin_fram_par.second;               // = 3-tuple (A,W,b)
        auto act_fram_par = relu_deep(lin_fram_par.first); // send in Z. Get (A,activation_cache=Z) 
        A = act_fram_par.first;
        cache = std::make_pair(linear_cache, act_fram_par.second); // (linear_cache,activation_cache)
    }

    return {A, cache};
}

pair<Mat1d,vector<pair<tuple_3, Mat1d>>> forwardProp(const Mat1d &X, const mat1dMap &params){
   
    vector<pair<tuple_3, Mat1d>> caches;
    const int L = params.size() / 2; // divide by 2 to get nr Layers in network 

    Mat1d A = X;
    Mat1d A_prev;
    // First we do [LINEAR -> RELU]*(L-1). Save the "cache" to the "caches" list.
    // The for loop starts at 1 because layer 0 is the input
    for (int l = 1; l < L; ++l)
    {
        string W_name = Wstr + to_string(l);
        string b_name = bstr + to_string(l);

        A_prev = A;
        auto par = linear_activation_forward(A_prev,params.at(W_name),params.at(b_name),"relu");
        A = par.first; // par.first is "A" (a Mat1d) returned from linear_activation_forward
        // par.second is the cache from linear_activation_forward. 
        caches.push_back(par.second); // par.second=cache=(linear_cache,activation_cache) 
    }
    // Last forward step LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    string W_name = Wstr + to_string(L);
    string b_name = bstr + to_string(L);
    auto par = linear_activation_forward(A,params.at(W_name),params.at(b_name),"sigmoid");
    Mat1d AL = par.first;  // Activations of Last layer. i.e Y-hat
    caches.push_back(par.second); // save (linear_cache,activation_cache) from last layer.
    return {AL,caches}; // return a pair  
}


// **** BACKWARD PROPAGATION FUNCTIONS  ****   

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
mat1dMap backProp(const Mat1d &AL, const Mat1d &Y, const vector<pair<tuple_3, Mat1d>> &caches)
{
    mat1dMap grads;

    const int L = caches.size(); // nr of Layers in network 
  
    // AL is our matrix/vector of predictions/y_hat (dim_of_final output x m =nr of examples)
    Mat1d dAL = -((Y/AL) -( (1.0 - Y)/(1.0 - AL) )); // dAL term is dL/da deriv. of cost w.r.t final output/y_hat.

    pair<tuple_3,Mat1d> current_cache = caches.at(L-1);
    tuple_3 tp3 = linear_activation_backward(dAL,current_cache,"sigmoid");

    Mat1d dA_prev_tmp = std::get<0>(tp3);
    Mat1d dW_tmp = std::get<1>(tp3);
    Mat1d db_tmp = std::get<2>(tp3);

    grads.emplace((dA + to_string(L-1)),dA_prev_tmp);
    grads.emplace((dW + to_string(L)),dW_tmp);
    grads.emplace((db + to_string(L)),db_tmp);

    for (int l=L-2; l >= 0; --l) // we need to loop backwards from l=L-2 to l=0
    {
        pair<tuple_3,Mat1d> current_cache = caches.at(l);
        tuple_3 tp3 = linear_activation_backward(grads.at((dA + to_string(l+1))),current_cache,"relu");

        Mat1d dA_prev_tmp = std::get<0>(tp3);
        Mat1d dW_tmp = std::get<1>(tp3);
        Mat1d db_tmp = std::get<2>(tp3);

        grads.emplace((dA + to_string(l)),dA_prev_tmp);
        grads.emplace((dW + to_string(l+1)),dW_tmp);
        grads.emplace((db + to_string(l+1)),db_tmp);

    }
    return grads;
}

mat1dMap backProp_L2_regularization(const Mat1d &AL, const Mat1d &Y, const vector<pair<tuple_3, Mat1d>> &caches, double lambd)
{
    mat1dMap grads;
    const double m = Y.cols;

    const int L = caches.size(); // nr of Layers in network 

    // AL is our matrix/vector of predictions/y_hat (dim_of_final output "x" m = nr of examples)
    Mat1d dAL = -((Y/AL) -( (1.0 - Y)/(1.0 - AL) )); // dAL term is dL/da deriv. of cost w.r.t final output/y_hat.

    pair<tuple_3,Mat1d> current_cache = caches.at(L-1);

    // Before sending in the cache to linear_activation_backward we get the W param matrix.
    Mat1d W_L = std::get<1>(current_cache.first);// current_cache_first is linear_cache which is (A,W,b)

    tuple_3 tp3 = linear_activation_backward(dAL,current_cache,"sigmoid");

    Mat1d dA_prev_tmp = std::get<0>(tp3);
    Mat1d dW_tmp = std::get<1>(tp3) + (lambd/m)*W_L;
    Mat1d db_tmp = std::get<2>(tp3);

    grads.emplace((dA + to_string(L-1)),dA_prev_tmp);
    grads.emplace((dW + to_string(L)),dW_tmp);
    grads.emplace((db + to_string(L)),db_tmp);

    for (int l=L-2; l >= 0; --l) // we need to loop backwards from l=L-2 to l=0
    {
        pair<tuple_3,Mat1d> current_cache = caches.at(l);
        tuple_3 tp3 = linear_activation_backward(grads.at((dA + to_string(l+1))),current_cache,"relu");

        Mat1d W_l = std::get<1>(current_cache.first);   // W for layer l saved in the linear cache
        Mat1d dA_prev_tmp = std::get<0>(tp3);
        Mat1d dW_tmp = std::get<1>(tp3) + (lambd/m)* W_l;
        Mat1d db_tmp = std::get<2>(tp3);

        grads.emplace((dA + to_string(l)),dA_prev_tmp);
        grads.emplace((dW + to_string(l+1)),dW_tmp);
        grads.emplace((db + to_string(l+1)),db_tmp);

    }
    return grads;
}


// **** FUNCTIONS FOR UPDATING OF PARAMETERS ****   

/**
 * @brief Update parameters using normal Gradient Descent without extra optimization methods
 * 
 * @param params map containing our parameters: e.g params['W' + str(l)] = Wl
 * @param grads map containing gradients for each parameters: e.g grads['dW' + str(l)] = dWl
 * @param LEARNING_RATE learning rate hyperparameter of Gradient Descent
 */
void update_parameters(mat1dMap &params, const mat1dMap &grads, const double LEARNING_RATE)
{
    const int L = params.size() / 2; // divide by 2 to get nr Layers in network 
 
    for (int l = 0; l < L; l++)
    {
        const string layer_str = to_string(l+1);
        string W_key = Wstr + layer_str;
        string b_key = bstr + layer_str;
        string dW_key = dW + layer_str;
        string db_key = db + layer_str;

        Mat1d W = params.at(W_key)  - (LEARNING_RATE * grads.at(dW_key));
        Mat1d b = params.at(b_key) - (LEARNING_RATE * grads.at(db_key)); 
    
        params.insert_or_assign(W_key,W); // could do params.erase(W_key); params.emplace(W_key,W);
        params.insert_or_assign(b_key,b);

    }  
}

/**
 * @brief Update parameters using Adam
 * 
 * @param params map containing our parameters: e.g params['W' + str(l)] = Wl
 * @param grads map containing gradients for each parameters: e.g grads['dW' + str(l)] = dWl
 * @param v_params Adam variable, moving average of the first gradient
 * @param s_params Adam variable, moving average of the squared gradient
 * @param t Adam variable, counts the number of taken steps
 * @param BETA1 Exponential decay hyperparameter for the first moment estimates 
 * @param BETA2 Exponential decay hyperparameter for the second moment estimates 
 * @param LEARNING_RATE learning rate hyperparameter of Gradient Descent
 */
void update_parameters_adam(mat1dMap &params, const mat1dMap &grads, mat1dMap &v_params, mat1dMap &s_params, const int t, const double BETA1,const double BETA2, const double LEARNING_RATE)
{
   const int L = params.size() / 2; // divide by 2 to get nr Layers in network 
   const double epsilon = 1e-8; // hyperparameter preventing division by zero in Adam updates used for numerical stability

   for (int l = 0; l < L; l++)
    {
        const string layer_str = to_string(l+1);
        string W_key = Wstr + layer_str; string b_key = bstr + layer_str;
        string dW_key = dW + layer_str; string db_key = db + layer_str;

        v_params.at(dW_key) = (BETA1*v_params.at(dW_key) + (1-BETA1)*grads.at(dW_key));      
        v_params.at(db_key) = (BETA1*v_params.at(db_key) + (1-BETA1)*grads.at(db_key)); 

        Mat1d v_dW_corrected = v_params.at(dW_key) / (1-std::pow(BETA1,t)); 
        Mat1d v_db_corrected = v_params.at(db_key) / (1-std::pow(BETA1,t));

        Mat1d grads_dW_squared;
        Mat1d grads_db_squared;
        cv::pow(grads.at(dW_key),2.0,grads_dW_squared);
        cv::pow(grads.at(db_key),2.0,grads_db_squared);

        s_params.at(dW_key) = (BETA2*s_params.at(dW_key) + (1-BETA2)*grads_dW_squared); 
        s_params.at(db_key) = (BETA2*s_params.at(db_key) + (1-BETA2)*grads_db_squared);

        Mat1d s_dW_corrected = s_params.at(dW_key)  / (1-std::pow(BETA2,t));     
        Mat1d s_db_corrected = s_params.at(db_key) / (1-std::pow(BETA2,t));

        Mat1d sp_dW_corr_sqrt;
        Mat1d sp_db_corr_sqrt;
        cv::sqrt(s_dW_corrected, sp_dW_corr_sqrt);
        cv::sqrt(s_db_corrected, sp_db_corr_sqrt);

  
        Mat1d W = params.at(W_key)  - (LEARNING_RATE*(v_dW_corrected / (sp_dW_corr_sqrt + epsilon)));
        Mat1d b = params.at(b_key) -  (LEARNING_RATE*(v_db_corrected / (sp_db_corr_sqrt + epsilon)));

        params.insert_or_assign(W_key,W); // could do params.erase(W_key); params.emplace(W_key,W);
        params.insert_or_assign(b_key,b);

    }
}





double predictAndCalcAccuracyDeep(const cv::Mat1d &X, const cv::Mat1d &Y, const mat1dMap &params)
{
    const int m = X.cols;

    Mat1d p = Mat::zeros(1, m, CV_64F);
    int numCorrect = 0;
    auto output =  forwardProp(X, params); // returns a pair (mat1d,vector of caches)
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
    return (accuracy * 100);     
}


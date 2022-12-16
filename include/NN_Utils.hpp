#ifndef NN_UTILS
#define NN_UTILS
#include <vector>
#include <utility>
#include <opencv2/core.hpp>

// Type aliases used
using mat1dMap = std::map<std::string, cv::Mat1d>;
using tuple_3  = std::tuple<cv::Mat1d,cv::Mat1d,cv::Mat1d>;

// Common functions used in both type of networks 
void printMap(const mat1dMap &params);
void printMatSlice(const cv::Mat1d& matrix, int rowsToPrint, int colsToPrint);

// Common functions used in both type of networks 
std::vector<int> layer_sizes(const cv::Mat1d &X, const cv::Mat1d &Y);
void initialize_parameters(const std::vector<int> &layer_dims, mat1dMap &params);

void initialize_adam(const mat1dMap &params, mat1dMap &v_params, mat1dMap &s_params);

double compute_cost_mse(const cv::Mat1d &AL, const cv::Mat1d &Y); 
double compute_cost_ce (const cv::Mat1d &AL, const cv::Mat1d &Y);
double compute_cost_ce_L2_reg(const cv::Mat1d &AL, const cv::Mat1d &Y,const mat1dMap &params, double lambd);


std::pair<cv::Mat1d,cv::Mat1d> sigmoid_deep(const cv::Mat1d& Z);
std::pair<cv::Mat1d,cv::Mat1d>relu_deep(const cv::Mat1d &Z);
std::pair<cv::Mat1d, cv::Mat1d> tanh_deep(const cv::Mat1d &Z);

cv::Mat1d sigmoid_backward(const cv::Mat1d& dA, const cv::Mat1d& cache);
cv::Mat1d relu_backward(const cv::Mat1d& dA, const cv::Mat1d& cache);
cv::Mat1d tanh_backward(const cv::Mat1d &dA, const cv::Mat1d &cache);



std::pair<cv::Mat1d,tuple_3> linear_forward(const cv::Mat1d& A,const cv::Mat1d& W,const cv::Mat1d& b);

std::pair<cv::Mat1d, std::pair<tuple_3, cv::Mat1d>> linear_activation_forward(const cv::Mat1d &A_prev, const cv::Mat1d &W, const cv::Mat1d &b, std::string activation);

std::pair<cv::Mat1d,std::vector<std::pair<tuple_3, cv::Mat1d>>> forwardProp(const cv::Mat1d &X, const mat1dMap &params);


tuple_3 linear_backward(const cv::Mat1d &dZ, tuple_3 linear_cache);
tuple_3 linear_activation_backward(const cv::Mat1d &dA, const std::pair<tuple_3, cv::Mat1d> &cache, std::string activation);

mat1dMap backProp(const cv::Mat1d &AL, const cv::Mat1d &Y, const std::vector<std::pair<tuple_3, cv::Mat1d>> &caches);
mat1dMap backProp_L2_regularization(const cv::Mat1d &AL, const cv::Mat1d &Y, const std::vector<std::pair<tuple_3, cv::Mat1d>> &caches,double lambd);

void update_parameters(mat1dMap &params, const mat1dMap &grads, const double learning_rate); 
void update_parameters_adam(mat1dMap &params, const mat1dMap &grads, mat1dMap &v_params, mat1dMap &s_params, const int counter, const double BETA1, const double BETA2, const double LEARNING_RATE);

double predictAndCalcAccuracyDeep(const cv::Mat1d &X, const cv::Mat1d &Y, const mat1dMap &params);
#endif

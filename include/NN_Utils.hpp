#ifndef NN_UTILS
#define NN_UTILS
#include <vector>
#include <opencv2/core.hpp>

using mat1dMap = std::map<std::string, cv::Mat1d>;

cv::Mat1d sigmoid(const cv::Mat1d &data);
cv::Mat1d sigmoid_derivative(const cv::Mat1d &data);

cv::Mat1d tanh(const cv::Mat1d &data);
cv::Mat1d tanh_derivative(const cv::Mat1d &data);

cv::Mat1d relu(const cv::Mat1d &data);
cv::Mat1d relu_derivative(const cv::Mat1d &data);

double l2_loss(const cv::Mat1d &yhat, const cv::Mat1d &y);

std::vector<int> layer_sizes(const cv::Mat1d &X, const cv::Mat1d &Y);

void initialize_parameters(const std::vector<int> &layer_dims, mat1dMap &params);

mat1dMap forward_propagation(const cv::Mat1d &X, const mat1dMap &params);

double compute_cost(const cv::Mat1d &Y_hat, const cv::Mat1d &Y);

mat1dMap back_prop(const mat1dMap &params, const mat1dMap &cache, const cv::Mat1d &X, const cv::Mat1d &Y);

mat1dMap &update_parameters(mat1dMap &params, mat1dMap &grads, double learning_rate);

cv::Mat1d predict(const cv::Mat1d &X, const mat1dMap &params);

void calcAndPrintAccuracy(const cv::Mat1d &predictions, const cv::Mat1d &Y);
#endif

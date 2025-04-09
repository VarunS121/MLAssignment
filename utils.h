#ifndef UTILS_H
#define UTILS_H

#include <vector>

// float randomWeight();
float sigmoid(float x);
float sigmoid_derivative(float x);
float relu(float x);
float relu_deriv(float x);
float crossEntropyLoss(const std::vector<float>& pred, int label);
std::vector<float> softmax(const std::vector<float>& input);
int argmax(const std::vector<float>& v);

extern float learning_rate;

#endif

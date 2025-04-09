#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>

float learning_rate = 0.001f;

// float randomWeight() {
//   // Xavier initialization range
//   // return ((float)rand() / RAND_MAX - 0.5f) * 2.0f * 0.05f;
//   std::random_device rd;
//   std::mt19937 gen(rd());  // Mersenne Twister engine
//   std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
//   float rand_value = dist(gen);
//   return rand_value;
// }

// Sigmoid activation
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float sigmoid_derivative(float x) {
  float s = sigmoid(x);
  return s * (1.0f - s);
}

float relu(float x) { return std::max(0.0f, x); }

float relu_deriv(float x) { return x > 0.0f ? 1.0f : 0.0f; }

std::vector<float> softmax(const std::vector<float>& input) {
  std::vector<float> output(input.size());
  float max_val = *std::max_element(input.begin(), input.end());

  float sum = 0.0f;
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = std::exp(input[i] - max_val);  // for numerical stability
    sum += output[i];
  }

  for (float& val : output) val /= sum;

  return output;
}

float crossEntropyLoss(const std::vector<float>& pred, int label) {
  float epsilon = 1e-9f;
  return -std::log(pred[label] + epsilon);
}

int argmax(const std::vector<float>& v) {
  return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

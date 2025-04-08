// // File: cnn.cpp
// #include "cnn.h"
// #include "utils.h"
// #include <vector>
// #include <cmath>
// #include <iostream>
// #include <random>
// #include <bits/stdc++.h>

// // ---------------- Layer Definitions ----------------

// struct Conv2D {
//     int in_channels, out_channels, kernel_size;
//     vector<vector<vector<vector<float>>>> weights; // [out][in][k][k]
//     vector<float> bias;

//     Conv2D(int in_c, int out_c, int k)
//         : in_channels(in_c), out_channels(out_c), kernel_size(k) {
//         weights.resize(out_c, vector<vector<vector<float>>>(in_c, vector<vector<float>>(k, vector<float>(k))));
//         bias.resize(out_c);
//         for (auto &oc : weights)
//             for (auto &ic : oc)
//                 for (auto &row : ic)
//                     for (auto &val : row)
//                         val = randomWeight();
//         for (auto &b : bias) b = randomWeight();
//     }

//     vector<vector<vector<float>>> forward(const vector<vector<vector<float>>> &input) {
//         int h = input[0].size();
//         int w = input[0][0].size();
//         int out_h = h - kernel_size + 1;
//         int out_w = w - kernel_size + 1;
//         vector<vector<vector<float>>> out(out_channels,
//             vector<vector<float>>(out_h, vector<float>(out_w, 0)));

//         for (int oc = 0; oc < out_channels; ++oc) {
//             for (int y = 0; y < out_h; ++y) {
//                 for (int x = 0; x < out_w; ++x) {
//                     float sum = bias[oc];
//                     for (int ic = 0; ic < in_channels; ++ic) {
//                         for (int ky = 0; ky < kernel_size; ++ky) {
//                             for (int kx = 0; kx < kernel_size; ++kx) {
//                                 sum += input[ic][y + ky][x + kx] * weights[oc][ic][ky][kx];
//                             }
//                         }
//                     }
//                     out[oc][y][x] = max(0.0f, sum); // ReLU
//                 }
//             }
//         }
//         return out;
//     }
// };

// struct MaxPool2D {
//     int size;
//     MaxPool2D(int s = 2) : size(s) {}

//     vector<vector<vector<float>>> forward(const vector<vector<vector<float>>> &input) {
//         int c = input.size();
//         int h = input[0].size() / size;
//         int w = input[0][0].size() / size;
//         vector<vector<vector<float>>> out(c, vector<vector<float>>(h, vector<float>(w)));

//         for (int ch = 0; ch < c; ++ch) {
//             for (int i = 0; i < h; ++i) {
//                 for (int j = 0; j < w; ++j) {
//                     float max_val = -1e9;
//                     for (int m = 0; m < size; ++m)
//                         for (int n = 0; n < size; ++n)
//                             max_val = max(max_val, input[ch][i * size + m][j * size + n]);
//                     out[ch][i][j] = max_val;
//                 }
//             }
//         }
//         return out;
//     }
// };

// struct Dropout {
//     float rate;
//     Dropout(float r = 0.5f) : rate(r) {}
//     vector<float> apply(const vector<float> &input) {
//         vector<float> out = input;
//         return out; // Simulated: no-op for now
//     }
// };

// struct Dense {
//     int in_size, out_size;
//     vector<vector<float>> weights;
//     vector<float> bias;

//     Dense(int in_sz, int out_sz) : in_size(in_sz), out_size(out_sz) {
//         weights.resize(out_sz, vector<float>(in_sz));
//         bias.resize(out_sz);
//         for (auto &row : weights)
//             for (float &w : row)
//                 w = randomWeight();
//         for (auto &b : bias)
//             b = randomWeight();
//     }

//     vector<float> forward(const vector<float> &input) {
//         vector<float> out(out_size, 0.0f);
//         for (int i = 0; i < out_size; ++i) {
//             for (int j = 0; j < in_size; ++j)
//                 out[i] += weights[i][j] * input[j];
//             out[i] += bias[i];
//         }
//         return out;
//     }
// };

// // ---------------- Model Definition ----------------

// Conv2D conv1(3, 32, 3);
// Conv2D conv2(32, 32, 3);
// MaxPool2D pool1;
// Dropout drop1;

// Conv2D conv3(32, 64, 3);
// Conv2D conv4(64, 64, 3);
// MaxPool2D pool2;
// Dropout drop2;

// Dense dense1(2304, 512);
// Dropout drop3;
// Dense dense2(512, 10);

// // Flatten function
// vector<float> flatten(const vector<vector<vector<float>>> &input) {
//     vector<float> out;
//     for (const auto &channel : input)
//         for (const auto &row : channel)
//             for (float val : row)
//                 out.push_back(val);
//     return out;
// }

// // Softmax
// vector<float> softmax(const vector<float> &logits) {
//     vector<float> exp_vals(logits.size());
//     float max_logit = *max_element(logits.begin(), logits.end());
//     float sum = 0.0f;
//     for (int i = 0; i < logits.size(); ++i) {
//         exp_vals[i] = exp(logits[i] - max_logit);
//         sum += exp_vals[i];
//     }
//     for (float &v : exp_vals) v /= sum;
//     return exp_vals;
// }

// // Predict (forward pass)
// vector<float> cnnPredict(const Image &img) {
//     // Normalize input
//     vector<vector<vector<float>>> input(3, vector<vector<float>>(32, vector<float>(32)));
//     for (int c = 0; c < 3; ++c)
//         for (int i = 0; i < 32; ++i)
//             for (int j = 0; j < 32; ++j)
//                 input[c][i][j] = img.data[c][i][j] / 255.0f;

//     auto x = conv1.forward(input);
//     x = conv2.forward(x);
//     x = pool1.forward(x);
//     x = conv3.forward(x);
//     x = conv4.forward(x);
//     x = pool2.forward(x);
//     auto flat = flatten(x);
//     auto h1 = dense1.forward(flat);
//     for (auto &v : h1) v = max(0.0f, v); // ReLU
//     auto out = dense2.forward(h1);
//     return softmax(out);
// }

// // Return index of highest probability
// // int argmax(const vector<float> &v) {
// //     return distance(v.begin(), max_element(v.begin(), v.end()));
// // }

// // Training stub
// void cnnTrain(const vector<Image> &train_data, int epochs) {
//     cout << "Training not yet implemented for full CNN. Forward pass ready.\n";
// }

// File: cnn.cpp
#include "cnn.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

// float learning_rate = 0.01f;

// float relu(float x) {
//     return std::max(0.0f, x);
// }
// float relu_deriv(float x) {
//     return x > 0 ? 1.0f : 0.0f;
// }

// std::vector<float> softmax(const std::vector<float>& logits) {
//     std::vector<float> result(logits.size());
//     float max_logit = *std::max_element(logits.begin(), logits.end());
//     float sum = 0.0f;
//     for (size_t i = 0; i < logits.size(); ++i) {
//         result[i] = std::exp(logits[i] - max_logit);
//         sum += result[i];
//     }
//     for (float& val : result) val /= sum;
//     return result;
// }

// int argmax(const std::vector<float>& v) {
//     return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
// }

// float crossEntropyLoss(const std::vector<float>& predicted, int true_label) {
//     return -std::log(std::max(predicted[true_label], 1e-7f));
// }

std::vector<float> one_hot(int label, int size = 10) {
    std::vector<float> out(size, 0.0f);
    out[label] = 1.0f;
    return out;
}

struct Conv2D {
  int in_channels, out_channels, kernel_size;
  std::vector<std::vector<std::vector<std::vector<float>>>> weights; // [out][in][k][k]
  std::vector<float> bias;
  std::vector<std::vector<std::vector<float>>> input_cache;

  Conv2D(int in_c, int out_c, int k)
      : in_channels(in_c), out_channels(out_c), kernel_size(k) {
      weights.resize(out_c, std::vector<std::vector<std::vector<float>>>(in_c,
                std::vector<std::vector<float>>(k, std::vector<float>(k))));
      bias.resize(out_c);
      for (auto& oc : weights)
          for (auto& ic : oc)
              for (auto& row : ic)
                  for (auto& val : row)
                      val = randomWeight();
      for (auto& b : bias) b = randomWeight();
  }

  std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
      input_cache = input;
      int h = input[0].size(), w = input[0][0].size();
      int out_h = h - kernel_size + 1;
      int out_w = w - kernel_size + 1;

      std::vector<std::vector<std::vector<float>>> out(out_channels,
          std::vector<std::vector<float>>(out_h, std::vector<float>(out_w)));

      for (int oc = 0; oc < out_channels; ++oc)
          for (int y = 0; y < out_h; ++y)
              for (int x = 0; x < out_w; ++x) {
                  float sum = bias[oc];
                  for (int ic = 0; ic < in_channels; ++ic)
                      for (int ky = 0; ky < kernel_size; ++ky)
                          for (int kx = 0; kx < kernel_size; ++kx)
                              sum += input[ic][y + ky][x + kx] * weights[oc][ic][ky][kx];
                  out[oc][y][x] = relu(sum);
              }
      return out;
  }

  std::vector<std::vector<std::vector<float>>> backward(
      const std::vector<std::vector<std::vector<float>>>& grad_output) {

      int h = input_cache[0].size();
      int w = input_cache[0][0].size();
      int out_h = h - kernel_size + 1;
      int out_w = w - kernel_size + 1;

      std::vector<std::vector<std::vector<float>>> grad_input(in_channels,
          std::vector<std::vector<float>>(h, std::vector<float>(w, 0.0f)));

      for (int oc = 0; oc < out_channels; ++oc)
          for (int y = 0; y < out_h; ++y)
              for (int x = 0; x < out_w; ++x) {
                  float grad_val = grad_output[oc][y][x];
                  for (int ic = 0; ic < in_channels; ++ic)
                      for (int ky = 0; ky < kernel_size; ++ky)
                          for (int kx = 0; kx < kernel_size; ++kx) {
                              int in_y = y + ky;
                              int in_x = x + kx;
                              float inp = input_cache[ic][in_y][in_x];
                              float d_relu = relu_deriv(inp * weights[oc][ic][ky][kx]);
                              grad_input[ic][in_y][in_x] += grad_val * weights[oc][ic][ky][kx] * d_relu;

                              float delta_w = grad_val * inp;
                              weights[oc][ic][ky][kx] -= learning_rate * delta_w;
                          }
                  bias[oc] -= learning_rate * grad_val;
              }
      return grad_input;
  }
};

struct MaxPool2D {
    int pool_size;
    std::vector<std::vector<std::vector<std::pair<int, int>>>> max_indices;

    MaxPool2D(int size = 2) : pool_size(size) {}

    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& input) {
        int c = input.size();
        int h = input[0].size() / pool_size;
        int w = input[0][0].size() / pool_size;
        max_indices.resize(c, std::vector<std::vector<std::pair<int, int>>>(h, std::vector<std::pair<int, int>>(w)));
        std::vector<std::vector<std::vector<float>>> output(c, std::vector<std::vector<float>>(h, std::vector<float>(w)));
        
        std::cout << c << " " << h << " "<< w << "\n";
        for (int ch = 0; ch < c; ++ch)
            for (int i = 0; i < h; ++i)
                for (int j = 0; j < w; ++j) {
                    float max_val = -1e9;
                    int max_i = -1, max_j = -1;
                    for (int pi = 0; pi < pool_size; ++pi)
                        for (int pj = 0; pj < pool_size; ++pj) {
                            int r = i * pool_size + pi;
                            int s = j * pool_size + pj;
                            if (input[ch][r][s] > max_val) {
                                max_val = input[ch][r][s];
                                max_i = r;
                                max_j = s;
                            }
                        }
                    output[ch][i][j] = max_val;
                    max_indices[ch][i][j] = { max_i, max_j };
                }
        return output;
    }

    std::vector<std::vector<std::vector<float>>> backward(
        const std::vector<std::vector<std::vector<float>>>& grad_output) {
        int c = grad_output.size();
        int h = grad_output[0].size() * pool_size;
        int w = grad_output[0][0].size() * pool_size;

        std::vector<std::vector<std::vector<float>>> grad_input(c, std::vector<std::vector<float>>(h, std::vector<float>(w, 0.0f)));
		std::cout << "pool back started\n";
        for (int ch = 0; ch < c; ++ch)
            for (long unsigned int i = 0; i < grad_output[0].size(); ++i)
                for (long unsigned int j = 0; j < grad_output[0][0].size(); ++j) {
					std::cout << "loop: " << ch << " " << i << " " << j << "\n";
					auto [r, s] = max_indices[ch][i][j];
					std::cout << r << " " << s << "\n";
					grad_input[ch][r % grad_output[0].size()][s% grad_output[0][0].size()] = grad_output[ch][i][j];
					std::cout << "loop done\n";
                }

        return grad_input;
    }
};

std::vector<float> flatten(const std::vector<std::vector<std::vector<float>>>& input) {
    std::vector<float> out;
    for (const auto& ch : input)
        for (const auto& row : ch)
            for (float val : row)
                out.push_back(val);
    return out;
}

std::vector<std::vector<std::vector<float>>> unflatten(const std::vector<float>& flat, int c, int h, int w) {
    std::vector<std::vector<std::vector<float>>> out(c, std::vector<std::vector<float>>(h, std::vector<float>(w)));
    int idx = 0;
    for (int i = 0; i < c; ++i)
        for (int j = 0; j < h; ++j)
            for (int k = 0; k < w; ++k)
                out[i][j][k] = flat[idx++];
    return out;
}

struct Dense {
    int in_dim, out_dim;
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    std::vector<float> input_cache;

    Dense(int in_d, int out_d) : in_dim(in_d), out_dim(out_d) {
        weights.resize(out_d, std::vector<float>(in_d));
        bias.resize(out_d);
        for (auto& row : weights)
            for (float& w : row)
                w = randomWeight();
        for (float& b : bias)
            b = randomWeight();
    }

    std::vector<float> forward(const std::vector<float>& input) {
        input_cache = input;
        std::vector<float> out(out_dim);
        for (int i = 0; i < out_dim; ++i) {
            out[i] = bias[i];
            for (int j = 0; j < in_dim; ++j)
                out[i] += weights[i][j] * input[j];
        }
        return out;
    }

    std::vector<float> backward(const std::vector<float>& grad_output) {
        std::vector<float> grad_input(in_dim, 0.0f);
        for (int i = 0; i < out_dim; ++i)
            for (int j = 0; j < in_dim; ++j) {
                grad_input[j] += grad_output[i] * weights[i][j];
                weights[i][j] -= learning_rate * grad_output[i] * input_cache[j];
            }
        for (int i = 0; i < out_dim; ++i)
            bias[i] -= learning_rate * grad_output[i];
        return grad_input;
    }
};

// Global CNN layers based on the specified architecture
Conv2D conv1(3, 32, 3);
Conv2D conv2(32, 32, 3);
MaxPool2D pool1(2);

Conv2D conv3(32, 64, 3);
Conv2D conv4(64, 64, 3);
MaxPool2D pool2(2);

Dense fc1(6 * 6 * 64, 512);
Dense fc2(512, 10);

// Forward pass
std::vector<float> cnnForward(const Image& img) {
	auto input = img.to3D();  // Shape: [3][32][32]
    auto x = conv1.forward(input);
    x = conv2.forward(x);
    x = pool1.forward(x);
    x = conv3.forward(x);
    x = conv4.forward(x);
    x = pool2.forward(x);

    auto flat = flatten(x);
    
	auto dense1 = fc1.forward(flat);
    for (auto& v : dense1) v = relu(v);

    auto dense2 = fc2.forward(dense1);

	return softmax(dense2);
}

// Backward pass
void cnnBackward(const std::vector<float>& pred, int label) {
    std::vector<float> grad_softmax = pred;
    grad_softmax[label] -= 1.0f;

    std::cout << "AllBack started\n";	
	auto grad_fc2 = fc2.backward(grad_softmax);
    for (auto& v : grad_fc2) v *= relu_deriv(v);
    auto grad_fc1 = fc1.backward(grad_fc2);

    auto grad_flat = unflatten(grad_fc1, 64, 6, 6);
	std::cout << "FlatBack done\n";
    auto grad_pool2 = pool2.backward(grad_flat);
	std::cout << "pool2Back done\n";
    auto grad_conv4 = conv4.backward(grad_pool2);
	std::cout << "conv4Back done\n";
    auto grad_conv3 = conv3.backward(grad_conv4);
	std::cout << "conv3Back done\n";

    auto grad_pool1 = pool1.backward(grad_conv3);
	std::cout << "pool1Back done\n";
    auto grad_conv2 = conv2.backward(grad_pool1);
	std::cout << "conv2Back done\n";
    conv1.backward(grad_conv2);
	std::cout << "conv1Back done\n";
}

// Training function
void cnnTrainExample(std::vector<Image>& images, int epochs) {
    std::cout << "CNN started\n";
	for (int e = 0; e < epochs; ++e) {
        std::cout << "Epoch: " << e+1 << "/10\n";
		float loss_sum = 0.0f;
        int correct = 0;

        std::shuffle(images.begin(), images.end(), std::mt19937{ std::random_device{}() });
		std::cout << "Shuffle done\n";
        for (const auto& img : images) {
            auto pred = cnnForward(img);
            loss_sum += crossEntropyLoss(pred, img.label);
			std::cout << "all forward and CEL done done\n";

            if (argmax(pred) == img.label) correct++;
            cnnBackward(pred, img.label);
			std::cout << "backward done\n";
        }

        float acc = 100.0f * correct / images.size();
        std::cout << "Epoch " << e + 1 << ": Loss = " << loss_sum / images.size() << ", Accuracy = " << acc << "%\n";
    }
}

// Public forward API for prediction
std::vector<float> cnnForwardExample(const Image& img) {
    return cnnForward(img);  // Returns vector of size 10
}

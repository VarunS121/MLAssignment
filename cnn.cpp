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
//         weights.resize(out_c, vector<vector<vector<float>>>(in_c,
//         vector<vector<float>>(k, vector<float>(k)))); bias.resize(out_c); for
//         (auto &oc : weights)
//             for (auto &ic : oc)
//                 for (auto &row : ic)
//                     for (auto &val : row)
//                         val = randomWeight();
//         for (auto &b : bias) b = randomWeight();
//     }

//     vector<vector<vector<float>>> forward(const vector<vector<vector<float>>>
//     &input) {
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
//                                 sum += input[ic][y + ky][x + kx] *
//                                 weights[oc][ic][ky][kx];
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

//     vector<vector<vector<float>>> forward(const vector<vector<vector<float>>>
//     &input) {
//         int c = input.size();
//         int h = input[0].size() / size;
//         int w = input[0][0].size() / size;
//         vector<vector<vector<float>>> out(c, vector<vector<float>>(h,
//         vector<float>(w)));

//         for (int ch = 0; ch < c; ++ch) {
//             for (int i = 0; i < h; ++i) {
//                 for (int j = 0; j < w; ++j) {
//                     float max_val = -1e9;
//                     for (int m = 0; m < size; ++m)
//                         for (int n = 0; n < size; ++n)
//                             max_val = max(max_val, input[ch][i * size + m][j
//                             * size + n]);
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
//     vector<vector<vector<float>>> input(3, vector<vector<float>>(32,
//     vector<float>(32))); for (int c = 0; c < 3; ++c)
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
//     cout << "Training not yet implemented for full CNN. Forward pass
//     ready.\n";
// }

// File: cnn.cpp
#include "cnn.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

#include "utils.h"

std::vector<float> one_hot(int label, int size = 10) {
  std::vector<float> out(size, 0.0f);
  out[label] = 1.0f;
  return out;
}

float randomWeight() {
  // Xavier initialization range
  // return ((float)rand() / RAND_MAX - 0.5f) * 2.0f * 0.05f;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-50,
                                          50);  // integers between -50 and 50
  return dist(gen) / 100.0f;
}

struct Conv2D {
  int in_channels, out_channels, kernel_size;
  std::vector<std::vector<std::vector<std::vector<double>>>>
      weights;  // [out][in][k][k]
  std::vector<double> bias;
  std::vector<std::vector<std::vector<float>>> input_cache;

  Conv2D(int in_c, int out_c, int k)
      : in_channels(in_c), out_channels(out_c), kernel_size(k) {
    weights.resize(out_c, std::vector<std::vector<std::vector<double>>>(
                              in_c, std::vector<std::vector<double>>(
                                        k, std::vector<double>(k))));
    bias.resize(out_c);
    for (int i = 0; i < out_c; i++)
      for (int j = 0; j < in_c; j++)
        for (int l = 0; l < k; l++)
          for (int m = 0; m < k; m++) {
            weights[i][j][l][m] = randomWeight();
          }
    for (auto& b : bias) b = randomWeight();
  }

  std::vector<std::vector<std::vector<float>>> forward(
      const std::vector<std::vector<std::vector<float>>>& input) {
    input_cache = input;
    int h = input[0].size(), w = input[0][0].size();
    int out_h = h - kernel_size + 1;
    int out_w = w - kernel_size + 1;

    // for (int i = 0; i < weights[0][0][0].size(); i++) {
    //   std::cout << weights[0][0][0][i] << " ";
    // }
    // cout << endl;

    // std::cout << input.size() << " " << h << " "<< w << "\n";
    std::vector<std::vector<std::vector<float>>> out(
        out_channels,
        std::vector<std::vector<float>>(out_h, std::vector<float>(out_w)));

    for (int oc = 0; oc < out_channels; ++oc)
      for (int y = 0; y < out_h; ++y)
        for (int x = 0; x < out_w; ++x) {
          float sum = bias[oc];
          for (int ic = 0; ic < in_channels; ++ic)
            for (int ky = 0; ky < kernel_size; ++ky)
              for (int kx = 0; kx < kernel_size; ++kx)
                sum += input[ic][y + ky][x + kx] * weights[oc][ic][ky][kx];
          out[oc][y][x] = sigmoid(sum);
        }

    // std::cout << "=================Conv Layer Weights===================\n";
    // for (int i = 0; i < weights.size(); i++) {
    //   for (int j = 0; j < weights[0].size(); j++) {
    //     for (int k = 0; k < weights[0][0].size(); k++) {
    //       for (int l = 0; l < weights[0][0][0].size(); l++) {
    //         std::cout << weights[i][j][k][l] << " ";
    //       }
    //       std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "=======================================================\n";
    return out;
  }

  std::vector<std::vector<std::vector<float>>> backward(
      const std::vector<std::vector<std::vector<float>>>& grad_output) {
    int h = input_cache[0].size();
    int w = input_cache[0][0].size();
    int out_h = h - kernel_size + 1;
    int out_w = w - kernel_size + 1;

    // std::cout << input_cache.size() << " " << h << " "<< w << "\n";

    std::vector<std::vector<std::vector<float>>> grad_input(
        in_channels,
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
                float d_relu =
                    sigmoid_derivative(inp * weights[oc][ic][ky][kx]);
                grad_input[ic][in_y][in_x] +=
                    grad_val * weights[oc][ic][ky][kx] * d_relu;

                float delta_w = grad_val * inp;
                weights[oc][ic][ky][kx] -= learning_rate * delta_w;
              }
          bias[oc] -= learning_rate * grad_val;
        }

    // std::cout << "=================Conv Layer Weights===================\n";
    // for (int i = 0; i < weights.size(); i++) {
    //   for (int j = 0; j < weights[0].size(); j++) {
    //     for (int k = 0; k < weights[0][0].size(); k++) {
    //       for (int l = 0; l < weights[0][0][0].size(); l++) {
    //         std::cout << weights[i][j][k][l] << " ";
    //       }
    //       std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "=======================================================\n";
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
    max_indices.resize(c, std::vector<std::vector<std::pair<int, int>>>(
                              h, std::vector<std::pair<int, int>>(w)));
    std::vector<std::vector<std::vector<float>>> output(
        c, std::vector<std::vector<float>>(h, std::vector<float>(w)));

    // std::cout << c << " " << h << " "<< w << "\n";
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
          // std::cout << max_i << " " << max_j << std::endl;
          output[ch][i][j] = max_val;
          max_indices[ch][i][j] = {max_i, max_j};
        }
    return output;
  }

  std::vector<std::vector<std::vector<float>>> backward(
      const std::vector<std::vector<std::vector<float>>>& grad_output) {
    int c = grad_output.size();
    int h = grad_output[0].size() * pool_size;
    int w = grad_output[0][0].size() * pool_size;
    // std::cout << c << " " << h << " "<< w << "\n";
    std::vector<std::vector<std::vector<float>>> grad_input(
        c, std::vector<std::vector<float>>(h, std::vector<float>(w, 0.0f)));
    // std::cout << "pool back started\n";
    for (int ch = 0; ch < c; ++ch)
      for (long unsigned int i = 0; i < grad_output[0].size(); ++i)
        for (long unsigned int j = 0; j < grad_output[0][0].size(); ++j) {
          // std::cout << "loop: " << ch << " " << i << " " << j << "\n";
          auto [r, s] = max_indices[ch][i][j];
          // std::cout << r << " " << s << "\n";
          grad_input[ch][r % grad_output[0].size()]
                    [s % grad_output[0][0].size()] = grad_output[ch][i][j];
          // std::cout << "loop done\n";
        }

    return grad_input;
  }
};

class Dropout {
 public:
  float dropout_rate;
  bool is_training;

  std::vector<bool> mask1D;
  std::vector<std::vector<std::vector<bool>>> mask3D;

  Dropout(float rate) : dropout_rate(rate), is_training(true) {}

  void set_training(bool training) { is_training = training; }

  // Overload for 1D
  std::vector<float> forward(const std::vector<float>& input) {
    if (!is_training || dropout_rate <= 0.0f) return input;

    std::vector<float> output(input.size());
    mask1D.resize(input.size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < input.size(); ++i) {
      mask1D[i] = dist(gen) > dropout_rate;
      output[i] = mask1D[i] ? input[i] / (1.0f - dropout_rate) : 0.0f;
    }

    return output;
  }

  std::vector<float> backward(const std::vector<float>& grad_output) {
    if (!is_training || dropout_rate <= 0.0f) return grad_output;

    std::vector<float> grad_input(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
      grad_input[i] = mask1D[i] ? grad_output[i] / (1.0f - dropout_rate) : 0.0f;
    }

    return grad_input;
  }

  // Overload for 3D
  std::vector<std::vector<std::vector<float>>> forward(
      const std::vector<std::vector<std::vector<float>>>& input) {
    if (!is_training || dropout_rate <= 0.0f) return input;

    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    mask3D.resize(C, std::vector<std::vector<bool>>(H, std::vector<bool>(W)));
    auto output = input;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          bool keep = dist(gen) > dropout_rate;
          mask3D[c][h][w] = keep;
          output[c][h][w] =
              keep ? input[c][h][w] / (1.0f - dropout_rate) : 0.0f;
        }
      }
    }

    return output;
  }

  std::vector<std::vector<std::vector<float>>> backward(
      const std::vector<std::vector<std::vector<float>>>& grad_output) {
    if (!is_training || dropout_rate <= 0.0f) return grad_output;

    int C = grad_output.size();
    int H = grad_output[0].size();
    int W = grad_output[0][0].size();

    auto grad_input = grad_output;

    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          grad_input[c][h][w] =
              mask3D[c][h][w] ? grad_output[c][h][w] / (1.0f - dropout_rate)
                              : 0.0f;
        }
      }
    }

    return grad_input;
  }
};

std::vector<float> flatten(
    const std::vector<std::vector<std::vector<float>>>& input) {
  std::vector<float> out;
  for (const auto& ch : input)
    for (const auto& row : ch)
      for (float val : row) out.push_back(val);
  return out;
}

std::vector<std::vector<std::vector<float>>> unflatten(
    const std::vector<float>& flat, int c, int h, int w) {
  std::vector<std::vector<std::vector<float>>> out(
      c, std::vector<std::vector<float>>(h, std::vector<float>(w)));
  int idx = 0;
  for (int i = 0; i < c; ++i)
    for (int j = 0; j < h; ++j)
      for (int k = 0; k < w; ++k) out[i][j][k] = flat[idx++];
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
      for (float& w : row) w = randomWeight();
    for (float& b : bias) b = randomWeight();
  }

  std::vector<float> forward(const std::vector<float>& input) {
    input_cache = input;
    std::vector<float> out(out_dim);
    for (int i = 0; i < out_dim; ++i) {
      out[i] = bias[i];
      for (int j = 0; j < in_dim; ++j) out[i] += weights[i][j] * input[j];
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
    for (int i = 0; i < out_dim; ++i) bias[i] -= learning_rate * grad_output[i];

    // std::cout << "=================DENSE Layer Weights===================\n";
    // for (int i = 0; i < weights.size(); i++) {
    //   for (int j = 0; j < weights[0].size(); j++) {
    //     std::cout << weights[i][j] << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "=======================================================\n";
    return grad_input;
  }
};

// Global CNN layers based on the specified architecture
Conv2D conv1(3, 32, 3);
Conv2D conv2(32, 32, 3);
MaxPool2D pool1(2);
Dropout drop1(0.25f);

Conv2D conv3(32, 64, 3);
Conv2D conv4(64, 64, 3);
MaxPool2D pool2(2);
Dropout drop2(0.25f);

Dense fc1(5 * 5 * 64, 512);
Dense fc2(512, 10);

// Forward pass
std::vector<float> cnnForward(const Image& img) {
  drop1.set_training(true);
  drop2.set_training(true);

  auto input = img.to3D();  // Shape: [3][32][32]
  auto x = conv1.forward(input);
  x = conv2.forward(x);
  x = pool1.forward(x);
  x = drop1.forward(x);
  x = conv3.forward(x);
  x = conv4.forward(x);
  x = pool2.forward(x);
  x = drop2.forward(x);

  auto flat = flatten(x);

  auto dense1 = fc1.forward(flat);
  for (auto& v : dense1) v = sigmoid(v);

  auto dense2 = fc2.forward(dense1);

  return softmax(dense2);
}

// Backward pass
void cnnBackward(const std::vector<float>& pred, int label) {
  drop1.set_training(true);
  drop2.set_training(true);

  std::vector<float> grad_softmax = pred;
  grad_softmax[label] -= 1.0f;

  // std::cout << "AllBack started\n";
  auto grad_fc2 = fc2.backward(grad_softmax);
  for (auto& v : grad_fc2) v *= sigmoid_derivative(v);
  auto grad_fc1 = fc1.backward(grad_fc2);

  auto grad_flat = unflatten(grad_fc1, 64, 5, 5);
  // std::cout << "FlatBack done\n";
  auto grad_drop2 = drop2.backward(grad_flat);
  auto grad_pool2 = pool2.backward(grad_drop2);
  // std::cout << "pool2Back done\n";
  auto grad_conv4 = conv4.backward(grad_pool2);
  // std::cout << "conv4Back done\n";
  auto grad_conv3 = conv3.backward(grad_conv4);
  // std::cout << "conv3Back done\n";
  auto grad_drop1 = drop1.backward(grad_conv3);
  auto grad_pool1 = pool1.backward(grad_drop1);
  // std::cout << "pool1Back done\n";
  auto grad_conv2 = conv2.backward(grad_pool1);
  // std::cout << "conv2Back done\n";
  conv1.backward(grad_conv2);
  // std::cout << "conv1Back done\n";
}

// Training function
void cnnTrainExample(std::vector<Image>& images, int epochs) {
  std::cout << "CNN started\n";
  for (int e = 0; e < epochs; ++e) {
    std::cout << "Epoch: " << e + 1 << "/" << epochs << "\n";
    float loss_sum = 0.0f;
    int correct = 0;

    std::shuffle(images.begin(), images.end(),
                 std::mt19937{std::random_device{}()});
    // std::cout << "Shuffle done\n";
    float count = 0;
    float total = images.size();
    // std::cout << "Progress: " << (count/total) * 100 << "% ";
    for (const auto& img : images) {
      auto pred = cnnForward(img);
      loss_sum += crossEntropyLoss(pred, img.label);
      // std::cout << "all forward and CEL done done\n";

      if (argmax(pred) == img.label) correct++;
      cnnBackward(pred, img.label);
      count++;
      float progress = (count / total) * 100.0;
      std::cout << "Progress: " << progress << "%\r";
      // std::cout << "backward done\n";
    }
    std::cout << "\n";

    float acc = 100.0f * correct / images.size();
    std::cout << "Epoch " << e + 1 << ": Loss = " << loss_sum / images.size()
              << ", Accuracy = " << acc << "%\n";
  }
}

// Public forward API for prediction
std::vector<float> cnnForwardExample(const Image& img) {
  return cnnForward(img);  // Returns vector of size 10
}

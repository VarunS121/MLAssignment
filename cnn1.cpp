
// File: cnn.cpp

#include "cnn.h"
#include "utils.h"
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

using namespace std;

// ----- Helper Functions -----
float relu(float x) { return max(0.0f, x); }
float relu_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }

vector<float> softmax(const vector<float>& input) {
    float max_val = *max_element(input.begin(), input.end());
    vector<float> exp_vals(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        exp_vals[i] = exp(input[i] - max_val);
        sum += exp_vals[i];
    }
    for (float& val : exp_vals) val /= sum;
    return exp_vals;
}

int argmax(const vector<float>& vec) {
    return static_cast<int>(max_element(vec.begin(), vec.end()) - vec.begin());
}

float cross_entropy(const vector<float>& pred, int label) {
    return -log(max(pred[label], 1e-10f));
}

float randomWeight() {
    static random_device rd;
    static mt19937 gen(rd());
    static normal_distribution<float> d(0, 0.01);
    return d(gen);
}

// ----- Layer: Dense -----
class Dense {
public:
    vector<vector<float>> weights;
    vector<float> biases;
    vector<float> input, output, grad_output;
    int in_size, out_size;

    Dense(int in_size, int out_size) : in_size(in_size), out_size(out_size) {
        weights.resize(out_size, vector<float>(in_size));
        biases.resize(out_size);
        for (auto& row : weights)
            for (auto& w : row)
                w = randomWeight();
        for (auto& b : biases)
            b = randomWeight();
    }

    vector<float> forward(const vector<float>& in) {
        input = in;
        output.resize(out_size);
        for (int i = 0; i < out_size; ++i) {
            output[i] = biases[i];
            for (int j = 0; j < in_size; ++j)
                output[i] += weights[i][j] * in[j];
        }
        return output;
    }

    vector<float> backward(const vector<float>& grad_out, float lr) {
        grad_output = grad_out;
        vector<float> grad_input(in_size, 0.0f);
        for (int i = 0; i < out_size; ++i) {
            for (int j = 0; j < in_size; ++j) {
                grad_input[j] += weights[i][j] * grad_out[i];
                weights[i][j] -= lr * grad_out[i] * input[j];
            }
            biases[i] -= lr * grad_out[i];
        }
        return grad_input;
    }
};

// ----- Layer: Flatten -----
vector<float> flatten(const vector<vector<vector<float>>>& input) {
    vector<float> output;
    for (const auto& channel : input)
        for (const auto& row : channel)
            for (float val : row)
                output.push_back(val);
    return output;
}

// ----- CNN Class -----
class CNN {
public:
    Dense dense1, dense2;
    CNN() : dense1(6 * 6 * 64, 512), dense2(512, 10) {}

    vector<float> forward(const Image& img) {
        vector<vector<vector<float>>> dummy_input(3, vector<vector<float>>(32, vector<float>(32)));
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < 32; ++i)
                for (int j = 0; j < 32; ++j)
                    dummy_input[c][i][j] = img.pixels[c * 1024 + i * 32 + j] / 255.0f;

        vector<float> flat = flatten(dummy_input);
        vector<float> d1_out = dense1.forward(flat);
        for (float& x : d1_out) x = relu(x);
        vector<float> d2_out = dense2.forward(d1_out);
        return softmax(d2_out);
    }

    void train(const vector<Image>& images, int epochs, float lr) {
        for (int e = 0; e < epochs; ++e) {
            float loss = 0.0f;
            int correct = 0;
            for (const auto& img : images) {
                vector<float> pred = forward(img);
                loss += cross_entropy(pred, img.label);
                if (argmax(pred) == img.label)
                    correct++;

                vector<float> grad(10, 0.0f);
                for (int i = 0; i < 10; ++i)
                    grad[i] = pred[i] - (i == img.label ? 1.0f : 0.0f);

                vector<float> grad1 = dense2.backward(grad, lr);
                for (int i = 0; i < grad1.size(); ++i)
                    grad1[i] *= relu_derivative(dense1.output[i]);
                dense1.backward(grad1, lr);
            }
            cout << "Epoch " << e + 1 << " - Loss: " << loss / images.size()
                 << " - Accuracy: " << (float)correct / images.size() * 100 << "%" << endl;
        }
    }
};

static CNN cnn_model;

void trainCNN(const vector<Image>& data, int epochs, float lr) {
    cnn_model.train(data, epochs, lr);
}

vector<float> cnnPredict(const Image& img) {
    return cnn_model.forward(img);
}

#include "logreg.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <bits/stdc++.h>

LogisticRegression::LogisticRegression(int input_size, int num_classes)
    : input_size(input_size), num_classes(num_classes),
      weights(num_classes, std::vector<float>(input_size)),
      biases(num_classes, 0.0f) {
    for (auto& w : weights)
        std::generate(w.begin(), w.end(), randomWeight);
}

std::vector<float> LogisticRegression::predict_proba(const Image& img) const {
    std::vector<float> logits(num_classes, 0.0f);

    for (int c = 0; c < num_classes; ++c) {
        float z = biases[c];
        for (int i = 0; i < input_size; ++i) {
            z += weights[c][i] * (img.pixels[i] / 255.0f);
        }
        logits[c] = z;
    }

    return softmax(logits);
}

int LogisticRegression::predict(const Image& img) const {
    std::vector<float> probs = predict_proba(img);
    return argmax(probs);
}

void LogisticRegression::train(const std::vector<Image>& images, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch: " << epoch+1 << "/10\n";
		float count = 0;
        float total = images.size();
        for (const auto& img : images) {
            std::vector<float> probs = predict_proba(img);

            for (int c = 0; c < num_classes; ++c) {
                float error = ((img.label == c) ? 1.0f : 0.0f) - probs[c];

                for (int i = 0; i < input_size; ++i) {
                    weights[c][i] += learning_rate * error * (img.pixels[i] / 255.0f);
                }

                biases[c] += learning_rate * error;
            }
            count++;
            float progress = (count/total) * 100.0;
            std::cout << "Progress: " << progress << "%\r";
        }
        std::cout << std::endl;
    }
}

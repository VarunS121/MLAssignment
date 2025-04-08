#ifndef LOGREG_H
#define LOGREG_H

#include "image.h"
#include <vector>

class LogisticRegression {
public:
    LogisticRegression(int input_size, int num_classes = 10);

    std::vector<float> predict_proba(const Image& img) const;
    int predict(const Image& img) const;
    void train(const std::vector<Image>& images, int epochs);

private:
    int input_size;
    int num_classes;
    std::vector<std::vector<float>> weights;  // [class][input_dim]
    std::vector<float> biases;                // [class]
};

#endif // LOGREG_H

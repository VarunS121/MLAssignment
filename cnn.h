#ifndef CNN_H
#define CNN_H

#include "image.h"
#include <bits/stdc++.h>

// Trains the CNN for given images and number of epochs
void cnnTrainExample(std::vector<Image>& images, int epochs);

// Runs forward pass and returns probability for class 0 (for comparison)
std::vector<float> cnnForwardExample(const Image& img);

#endif

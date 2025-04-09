#include "image.h"

#include <cassert>
#include <fstream>
#include <iostream>

std::vector<std::string> loadLabelNames(const std::string& meta_filename) {
  std::ifstream file(meta_filename);
  std::vector<std::string> names;
  std::string line;
  while (std::getline(file, line)) {
    names.push_back(line);
  }
  return names;
}

std::vector<Image> loadCIFARBatch(const std::string& filename, int max_images) {
  std::ifstream file(filename, std::ios::binary);
  assert(file.is_open());

  std::vector<std::string> class_names =
      loadLabelNames("./cifar-10-batches-bin/batches.meta.txt");

  const int image_size = 3072;
  const int total_size = image_size + 1;

  std::vector<Image> images;

  for (int i = 0; i < max_images && file.peek() != EOF; ++i) {
    unsigned char label;
    file.read(reinterpret_cast<char*>(&label), 1);

    std::vector<unsigned char> pixels(image_size);
    file.read(reinterpret_cast<char*>(pixels.data()), image_size);

    Image img;
    img.label = static_cast<int>(label);
    img.pixels = pixels;
    img.class_name = class_names[img.label];
    images.push_back(img);
  }

  return images;
}

std::vector<std::vector<std::vector<float>>> Image::to3D() const {
  std::vector<std::vector<std::vector<float>>> result(
      3, std::vector<std::vector<float>>(32, std::vector<float>(32)));

  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 32; ++i) {
      for (int j = 0; j < 32; ++j) {
        result[c][i][j] = pixels[c * 1024 + i * 32 + j] / 255.0f;
      }
    }
  }

  return result;
}

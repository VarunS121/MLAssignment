#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

struct Image {
  std::vector<unsigned char> pixels;  // 3072 bytes = 32*32*3
  int label;
  std::string class_name;

  std::vector<std::vector<std::vector<float>>> to3D() const;
};

std::vector<Image> loadCIFARBatch(const std::string& filename,
                                  int max_images = 10000);
std::vector<std::string> loadLabelNames(const std::string& meta_filename);

#endif

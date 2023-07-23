#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <filesystem>
#include <vector>

//#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace utilities {

cv::Mat load_image_from_path(const std::filesystem::path& image_path, int w, int h);


}
//torch::Tensor make_tensor_stack(std::vector<torch::Tensor>& tensor);

#endif // UTILITIES_HPP
#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace cv;

cv::Mat load_image_from_path(const std::filesystem::path& image_path, int w, int h);



#endif // UTILITIES_HPP
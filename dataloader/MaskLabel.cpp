
#include "MaskLabel.hpp"

#include "utilities.hpp"

MaskLabel::MaskLabel(std::filesystem::path this_path) {
    this->path = this_path;
}

cv::Mat MaskLabel::load_image(int w, int h) const {
    return load_image_from_path(this->path,w,h);
}
#ifndef MASKLABEL_HPP
#define MASKLABEL_HPP

#include "Label.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>

class MaskLabel : public Label {
    
public:
    MaskLabel(std::filesystem::path this_path);
    cv::Mat load_image(int w, int h) const override;
    std::filesystem::path get_path() const { return path;}

private:
    std::filesystem::path path;
};


#endif // MASKLABEL_HPP
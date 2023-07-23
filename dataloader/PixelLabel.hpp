#ifndef PIXELLABEL_HPP
#define PIXELLABEL_HPP

#include <opencv2/opencv.hpp>

#include "Label.hpp"

using namespace cv;

class PixelLabel : public Label {
public:
    PixelLabel(int x, int y,int rad);
    cv::Mat load_image(int w, int h) const override;

private:
    int x;
    int y;
    int rad;
    cv::Mat generate_heatmap(int x, int y, const int rad, const int w, const int h) const;
};


#endif  // PIXELLABEL_HPP
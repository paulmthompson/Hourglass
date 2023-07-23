#ifndef LABEL_HPP
#define LABEL_HPP

#include <opencv2/opencv.hpp>

// Labels can take the form of 1) images (masks) or 2) pixel coordinates (x,y) which can be used to generate a heatmap image

class Label {

public:
    Label() = default;

    Label(const Label&) =delete;
    void operator=(const Label&) =delete;

    virtual ~Label() {}

    virtual cv::Mat load_image(int w, int h) const = 0;

private:
    cv::Mat resize_image(const cv::Mat& img, int w, int h) const;
};


#endif  // LABEL_HPP
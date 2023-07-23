
#include "Label.hpp"


cv::Mat Label::resize_image(const cv::Mat& raw_image, int w, int h) const {

    cv::Mat image;
    cv::resize(raw_image, image,cv::Size(w,h), cv::INTER_AREA);

    if (!image.isContinuous()) {   
        image = image.clone(); 
    }

    return image;
};

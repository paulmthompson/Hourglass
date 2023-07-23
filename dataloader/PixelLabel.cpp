
#include "PixelLabel.hpp"

#include <iostream>

PixelLabel::PixelLabel(int x, int y,int rad) {
    this->x = x;
    this->y = y;
    this->rad = rad; // This needs to be an odd number
}

cv::Mat PixelLabel::load_image(int w, int h) const {
    return this->generate_heatmap(this->x, this->y, this->rad, w, h);
}

 // This takes a 2D coordinate and generates a heatmap with a gaussian distribution
// The keypoint coordinate (x,y) is from an an image of size (img_w,img_h)
// The heatmap is generated with size (w,h)

cv::Mat PixelLabel::generate_heatmap(int x, int y, const int rad, const int w, const int h) const {
    
    int img_w = 640;
    int img_h = 480;

    if ((x > img_w) | (y > img_h)) {
        std::cout << "Coordinate is out of bounds" << std::endl;
    }

    cv::Mat raw_image = cv::Mat::zeros(img_h,img_w,CV_32FC1);
    if ((x >= 0) & (y >= 0)) {
        float & point = raw_image.at<float>(y,x);
        point = 255.0;
    } else {
        std::cout << "Label at (0,0) is interpreted as no label for this image" << std::endl;
    }

    cv::Mat raw_image2;
    cv::GaussianBlur(raw_image, raw_image2, cv::Size(rad,rad), 0);

    cv::Mat raw_image3;
    cv::normalize(raw_image2,raw_image3,0.0,255.0,cv::NORM_MINMAX);

    cv::Mat image = cv::Mat(h,w,CV_32FC1);

    cv::resize(raw_image3,image,image.size(),0.0,0.0,cv::INTER_AREA);

    cv::Mat image_out = cv::Mat(h,w,CV_8UC1);
    image.convertTo(image_out,CV_8UC1);

    if (!image_out.isContinuous()) {   
        image_out = image_out.clone(); 
    }

    //imwrite("test.png",image_out);

    return image_out;
};
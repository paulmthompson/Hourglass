#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <random>

using namespace torch;
namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace cv;

#pragma once

//Many strategies using OpenCv were adapted from this blog post
//https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5

void flip_images(std::vector<cv::Mat>& output_images, std::vector<cv::Mat>& output_labels, 
                const cv::Mat& input_image, const cv::Mat& input_label, const int flip_dir) {

    cv::Mat output_image = cv::Mat::zeros( input_image.size(), input_image.type() );
    cv::Mat output_label = cv::Mat::zeros( input_label.size(), input_label.type() );

    cv::flip(input_image,output_image,flip_dir);
    cv::flip(input_label,output_label,flip_dir);

    output_images.push_back(std::move(output_image));
    output_labels.push_back(std::move(output_label));
};

cv::Mat rotate_image(const cv::Mat& img, const int angle) {

    cv::Mat output_img = cv::Mat::zeros( img.size(), img.type() );

    int width = img.size().width;
    int height = img.size().height;

    auto M = cv::getRotationMatrix2D(cv::Point2f(width/2,height/2),angle,1);

    cv::warpAffine(img,output_img,M,cv::Size(width,height),
            cv::INTER_LINEAR,cv::BORDER_CONSTANT);

    return output_img;
};

void rotate_images(std::vector<cv::Mat>& output_images, std::vector<cv::Mat>& output_labels, 
                const cv::Mat& input_image, const cv::Mat& input_label, const int angle) {

    output_images.push_back(rotate_image(input_image,angle));
    output_labels.push_back(rotate_image(input_label,angle));

};

//https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
cv::Mat contrast_adjust(const cv::Mat& image, const float alpha, const int beta) {

    cv::Mat new_image = cv::Mat::zeros( image.size(), image.type() );

    image.convertTo(new_image,-1,alpha,beta);

    return new_image;
}

void contrast_adjust_images(std::vector<cv::Mat>& output_images, std::vector<cv::Mat>& output_labels, 
                const cv::Mat& input_image, const cv::Mat& input_label, const float alpha, const int beta) {
    
    output_images.push_back(contrast_adjust(input_image,alpha,beta));
    output_labels.push_back(input_label.clone());
};

cv::Mat horizontal_shift_image(const cv::Mat& image, const float ratio) { 
    cv::Mat new_image = cv::Mat::zeros( image.size(), image.type() );

    const int width = image.size().width;
    const int height = image.size().height;

    int to_shift = width * ratio;

    float warp_values[] = {1.0, 0.0, to_shift, 0.0, 1.0, 0.0};
    cv::Mat M = cv::Mat(2,3,CV_32F,warp_values);

    cv::warpAffine(image,new_image,M,cv::Size(width,height),
    cv::INTER_LINEAR,cv::BORDER_CONSTANT);

    return new_image;
};

void horizontal_shift_images(std::vector<cv::Mat>& output_images, std::vector<cv::Mat>& output_labels, 
                const cv::Mat& input_image, const cv::Mat& input_label, float ratio) {

    output_images.push_back(horizontal_shift_image(input_image,ratio));
    output_labels.push_back(horizontal_shift_image(input_label,ratio));
};

std::tuple<std::vector<cv::Mat>,std::vector<cv::Mat>> image_augmentation(const cv::Mat& input_image,const cv::Mat& input_label) {

    std::random_device seed;
    std::mt19937 gen{seed()}; // seed the generator
    

    std::vector<cv::Mat> output_images;
    std::vector<cv::Mat> output_labels;

    bool horizontal_flip = true;
    bool vertical_flip = true;

    bool rotate_image = true;
    std::uniform_int_distribution rotate{-30, 30};

    bool contrast_adjustment = true;
    std::uniform_real_distribution alpha{0.8, 3.0};
    std::uniform_int_distribution beta{-25, 100};

    bool horizontal_shift = true;
    std::uniform_real_distribution h_ratio{-0.2, 0.2};

    output_images.push_back(input_image.clone());
    output_labels.push_back(input_label.clone());

    if (horizontal_flip) {
        flip_images(output_images,output_labels,input_image,input_label,0);
    }
    if (vertical_flip) {
        flip_images(output_images,output_labels,input_image,input_label,1);
    }
    if (rotate_image) {
        rotate_images(output_images,output_labels,input_image,input_label,rotate(gen));
    }
    if (contrast_adjustment) {
        contrast_adjust_images(output_images,output_labels,input_image,input_label,alpha(gen),beta(gen));
    }
    if (horizontal_shift) {
        horizontal_shift_images(output_images,output_labels,input_image,input_label,h_ratio(gen));
    }

    return std::make_tuple(output_images,output_labels);
}
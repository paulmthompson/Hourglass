#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

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

    cv::warpAffine(img,output_img,M,cv::Size(width,height));

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

std::tuple<std::vector<cv::Mat>,std::vector<cv::Mat>> image_augmentation(const cv::Mat& input_image,const cv::Mat& input_label) {

    std::vector<cv::Mat> output_images;
    std::vector<cv::Mat> output_labels;

    bool horizontal_flip = true;
    bool vertical_flip = true;

    std::vector<int> rotations{10};

    bool contrast_adjustment = true;
    float alpha = 1.5;
    int beta = 50;

    output_images.push_back(input_image.clone());
    output_labels.push_back(input_label.clone());


    if (horizontal_flip) {
        flip_images(output_images,output_labels,input_image,input_label,0);
    }
    if (vertical_flip) {
        flip_images(output_images,output_labels,input_image,input_label,1);
    }
    for (auto& rotation : rotations) {
        rotate_images(output_images,output_labels,input_image,input_label,rotation);
    }
    if (contrast_adjustment) {
        contrast_adjust_images(output_images,output_labels,input_image,input_label,alpha,beta);
    }

    return std::make_tuple(output_images,output_labels);
}
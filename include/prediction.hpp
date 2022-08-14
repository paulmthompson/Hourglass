#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <ffmpeg_wrapper/videodecoder.h>

#include "hourglass.hpp"
#include "dataload.hpp"

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>  // for high_resolution_clock
#include <memory>

using namespace torch;
using json = nlohmann::json;

#pragma once

torch::Tensor prepare_for_opencv(torch::Tensor tensor,const int height, const int width) {

    tensor = nn::functional::interpolate(tensor,
        nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({height,width})).mode(torch::kNearest));

    tensor = tensor.mul(255).clamp(0,255).to(torch::kU8);

    tensor = tensor.detach().permute({2,3,1,0});
        
    return tensor.to(kCPU);
}

torch::Tensor get_hourglass_predictions(StackedHourglass &hourglass, torch::Tensor& data,const int height, const int width) {
    auto output = hourglass->forward(data);

    torch::Tensor prediction = output.back();
        
    return prepare_for_opencv(prediction,height, width);
}

cv::Mat combine_overlay(cv::Mat& img, cv::Mat& label) {
    cv::Mat color_img;
    cv::Mat channel[3];
    cv::cvtColor(img,color_img,cv::COLOR_GRAY2RGB);

    cv::split(color_img,channel);

    cv::addWeighted(channel[1],0.5, label,0.5,0.0,channel[1]);

    /*
    for (int i=0; i< color_img.rows; i++) {
        for (int j = 0; j< color_img.cols; j++) {

            float label_val = label.at<uint8_t>(i,j) / 255;

            uint8_t & red = channel[0].at<uint8_t>(i,j);
            uint8_t & green = channel[1].at<uint8_t>(i,j);
            uint8_t & blue = channel[2].at<uint8_t>(i,j);

            red = (uint8_t) (label_val * 255);
            //red = label_val + red * (255-label_val);
            //green = green * (255 - label_val);
            //blue = blue * (255 - label_val);

            channel[0].at<uint8_t>(i,j) = red;
            channel[1].at<uint8_t>(i,j) = green;
            channel[2].at<uint8_t>(i,j) = blue;
        }
    }
    */
    cv::merge(channel,3,color_img);
    return color_img;
}

template <class T>
void predict(StackedHourglass &hourglass, T &data_set, torch::Device device, const std::string &config_file)
{

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    hourglass->to(device);

    int batch_size = data["prediction"]["batch-size"];
    const int64_t batches_per_epoch = std::ceil(data_set.size().value() / static_cast<double>(batch_size));
    const int64_t total_images = data_set.size().value();

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(data_set),
        batch_size);

    load_weights(hourglass,config_file);

    const int out_height = 256;
    const int out_width = 256;

    auto start = std::chrono::high_resolution_clock::now();

    int64_t batch_index = 0;
    for (auto &batch : *data_loader)
    {
        auto data = batch.data.to(device);
        auto labels = batch.target.to(device);

        auto prediction = get_hourglass_predictions(hourglass,data,out_height,out_width);

        auto tensor_raw_data_ptr = prediction.data_ptr();

        for (int j = 0; j < prediction.size(3); j++) {

            cv::Mat resultImg(out_height,out_width,CV_8UC1, tensor_raw_data_ptr + (out_height*out_width*j));
            
            std::string img_name = "test" + std::to_string(batch_index) + "-" + std::to_string(j) + ".png";
            cv::imwrite(img_name,resultImg);
        }
      
        std::cout << "\r"
                    "["
                    << (++batch_index) * batch_size << "/" << total_images << "]"
                    << " Predicted" << std::flush;

    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - start;

    std::cout << std::endl;
    std::cout << total_images << " images predicted in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average " << total_images / elapsed.count() << " images per second" << std::endl;
};

void predict_video(StackedHourglass &hourglass, torch::Device device, const std::string &config_file)
{

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();
    
    hourglass->to(device);
    std::cout << data["prediction"]["videos"] << std::endl;

    auto vd = ffmpeg_wrapper::VideoDecoder();
     
    std::string vid_name = data["prediction"]["videos"];
    vd.createMedia(vid_name);
    const int64_t total_images = vd.getFrameCount();

    std::cout << "Video loaded with " << total_images << " frames" << std::endl;
    int batch_size = data["prediction"]["batch-size"];
    const int64_t batches_per_epoch = std::ceil(total_images / static_cast<double>(batch_size));
    
    load_weights(hourglass,config_file);

    const int out_height = vd.getHeight();
    const int out_width = vd.getWidth();

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> image = vd.getFrame(0, true);

    int64_t batch_index = 0;
    int64_t frame_index = 0;
    while (frame_index < total_images)
    {
        int last_index = frame_index + batch_size - 1;
        last_index = (last_index < total_images) ? last_index : total_images -1;
        auto data = LoadFrames(vd,frame_index,last_index);

        data = data.to(device);

        data = nn::functional::interpolate(data,
            nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({256,256})).mode(torch::kNearest));

        auto prediction = get_hourglass_predictions(hourglass,data,out_height,out_width);

        auto prediction_raw_data_ptr = prediction.data_ptr<uchar>();

        data = prepare_for_opencv(data,out_height,out_width);

        auto data_raw_data_ptr = data.data_ptr<uchar>();
        
        for (int j = 0; j < prediction.size(3); j++) {

            cv::Mat resultImg(out_height,out_width,CV_8UC1, prediction_raw_data_ptr + (out_height*out_width*j));
            cv::Mat realImg(out_height, out_width, CV_8UC1, data_raw_data_ptr + (out_height*out_width*j));

            resultImg = combine_overlay(realImg,resultImg);
            
            std::string img_name = "test" + std::to_string(batch_index) + "-" + std::to_string(j) + ".png";
            cv::imwrite(img_name,resultImg);
        }
        
        std::cout << "\r"
                    "["
                    << (++batch_index) * batch_size << "/" << total_images << "]"
                    << " Predicted" << std::flush;
        frame_index = frame_index + batch_size;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - start;

    std::cout << std::endl;
    std::cout << total_images << " images predicted in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average " << total_images / elapsed.count() << " images per second" << std::endl;

}
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


torch::Tensor get_hourglass_predictions(StackedHourglass &hourglass, torch::Tensor data,const int height, const int width) {
    auto output = hourglass->forward(data);

    torch::Tensor prediction = output.back();

    prediction = nn::functional::interpolate(prediction,
        nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({height,width})).mode(torch::kNearest));
        
    prediction = prediction.mul(255).clamp(0,255).to(torch::kU8);

    prediction = prediction.detach().permute({2,3,1,0});
        
    return prediction.to(kCPU);
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

        auto tensor_raw_data_ptr = prediction.data_ptr<uchar>();

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
}

void predict_video(StackedHourglass &hourglass, torch::Device device, const std::string &config_file)
{

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();
    
    //hourglass->to(device);
    std::cout << data["prediction"]["videos"] << std::endl;
    ffmpeg_wrapper::VideoDecoder vd = ffmpeg_wrapper::VideoDecoder(data["prediction"]["videos"]);

    //auto vd = std::make_unique<ffmpeg_wrapper::VideoDecoder>();
    
    /*
    vd->createMedia(data["prediction"]["videos"]);
    const int64_t total_images = vd->getFrameCount();
    int batch_size = data["prediction"]["batch-size"];
    const int64_t batches_per_epoch = std::ceil(total_images / static_cast<double>(batch_size));
    
    load_weights(hourglass,config_file);

    const int out_height = vd->getHeight();
    const int out_width = vd->getWidth();

    auto start = std::chrono::high_resolution_clock::now();

    int64_t batch_index = 0;
    int64_t frame_index = 0;
    while (frame_index < total_images)
    {
        int last_index = (frame_index + batches_per_epoch -1 < total_images) ? frame_index + batches_per_epoch - 1 : total_images -1;
        auto data = LoadFrames(vd,frame_index,last_index);

        auto prediction = get_hourglass_predictions(hourglass,data,out_height,out_width);

        auto tensor_raw_data_ptr = prediction.data_ptr<uchar>();
      
        std::cout << "\r"
                    "["
                    << (++batch_index) * batch_size << "/" << total_images << "]"
                    << " Predicted" << std::flush;
        frame_index = frame_index + batch_size - 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - start;

    std::cout << std::endl;
    std::cout << total_images << " images predicted in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average " << total_images / elapsed.count() << " images per second" << std::endl;
    */
}
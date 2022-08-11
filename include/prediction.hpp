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

using namespace torch;
using json = nlohmann::json;

#pragma once


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

        auto output = hourglass->forward(data);

        torch::Tensor prediction = output.back();

        prediction = nn::functional::interpolate(prediction,
            nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({out_height,out_width})).mode(torch::kNearest));
        
        prediction = prediction.mul(255).clamp(0,255).to(torch::kU8);

        prediction = prediction.detach().permute({2,3,1,0});
        
        prediction = prediction.to(kCPU);

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
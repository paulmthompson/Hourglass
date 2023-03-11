#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "hourglass.hpp"
#include "dataloader/dataload.hpp"

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace torch;
using json = nlohmann::json;

#pragma once

inline void write_loss(std::vector<float> loss) {

    std::ofstream wf("losses.dat", std::ios::out | std::ios::binary);

    if(!wf) {
      std::cout << "Cannot open file!" << std::endl;
   }

    wf.write(reinterpret_cast<const char *>(&loss[0]),sizeof(float)*loss.size());

    wf.close();

};

inline torch::Tensor intermediate_supervision(std::vector<torch::Tensor>& output, torch::Tensor& labels) {
    
    std::vector<torch::Tensor> losses;

    for (auto &level_output : output)
    {
        losses.push_back(torch::mse_loss(level_output, labels));
    }

    torch::Tensor loss = losses[0];
    for (int i = 1; i < losses.size(); i++)
    {
        loss += losses[i];
    }

    return loss;
};

template <class T>
inline void train_hourglass(StackedHourglass &hourglass, T &data_set, torch::Device device, training_options& options)
{
    if (options.load_weights) {
        load_weights(hourglass,options.load_weight_path);
    }
    hourglass->to(device);
    hourglass->train();

    const int64_t batches_per_epoch = std::ceil(data_set.size().value() / static_cast<double>(options.batch_size));

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(data_set),
        options.batch_size);

    std::vector<float> all_losses;

    torch::optim::Adam optimizer(
        hourglass->parameters(), torch::optim::AdamOptions(options.learning_rate).weight_decay(5e-4));

    std::cout << "Beginning Training for " << options.epochs << " Epochs" << std::endl;

    for (int64_t epoch = 1; epoch <= options.epochs; ++epoch)
    {
        int64_t batch_index = 0;

        for (auto &batch : *data_loader)
        {
            torch::AutoGradMode enable_grad(true);

            hourglass->zero_grad();

            auto data = batch.data.to(device);
            auto labels = batch.target.to(device);

            auto output = hourglass->forward(data);
            
            torch::Tensor loss;
            if (options.intermediate_supervision) {
                loss = intermediate_supervision(output,labels);
            } else {
                loss = torch::mse_loss(output.back(), labels);
            }

            loss.backward();

            optimizer.step();

            all_losses.push_back(loss.item<float>());

            std::cout << "\r"
                         "["
                      << epoch << "/" << options.epochs << "]"
                      << "[" << ++batch_index << "/" << batches_per_epoch << "]"
                      << " loss: " << loss.item<float>() << std::flush;
        }
    }

    hourglass->to(kFloat32);
    hourglass->to(torch::Device(torch::kCPU));
    torch::save(hourglass, options.weight_save_name);
    torch::save(optimizer, "hourglass-optimizer.pt");

    write_loss(all_losses);

}
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "hourglass.hpp"
#include "dataload.hpp"

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace torch;
using json = nlohmann::json;

#pragma once

void write_loss(std::vector<float> loss) {

    std::ofstream wf("losses.dat", std::ios::out | std::ios::binary);

    if(!wf) {
      std::cout << "Cannot open file!" << std::endl;
   }

    wf.write(reinterpret_cast<const char *>(&loss[0]),sizeof(float)*loss.size());

    wf.close();

};

template <class T>
void train_hourglass(StackedHourglass &hourglass, T &data_set, torch::Device device, training_options& options)
{

    hourglass->to(device);
    hourglass->train();

    const int64_t batches_per_epoch = std::ceil(data_set.size().value() / static_cast<double>(options.batch_size));

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(data_set),
        options.batch_size);

    std::vector<float> all_losses;

    torch::optim::Adam optimizer(
        hourglass->parameters(), torch::optim::AdamOptions(options.learning_rate).weight_decay(5e-4));

    load_weights(hourglass,options.config_file);

    std::cout << "Beginning Training for " << options.epochs << " Epochs" << std::endl;

    for (int64_t epoch = 1; epoch <= options.epochs; ++epoch)
    {
        int64_t batch_index = 0;
        for (auto &batch : *data_loader)
        {
            try {
                hourglass->zero_grad();
            } catch (const c10::Error &e) {
                std::cout << e.msg() << std::endl;
            }

            auto data = batch.data.to(device);
            auto labels = batch.target.to(device);

            auto output = hourglass->forward(data);
            
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

    torch::save(hourglass, options.weight_save_name);
    torch::save(optimizer, "hourglass-optimizer.pt");

    write_loss(all_losses);

}
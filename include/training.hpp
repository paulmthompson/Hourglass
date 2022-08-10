#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "hourglass.hpp"
#include "dataload.hpp"

#include <vector>
#include <cmath>

using namespace torch;
using json = nlohmann::json;

#pragma once

template <class T>
void train_hourglass(StackedHourglass &hourglass, T &data_set, torch::Device device, const std::string &config_file)
{

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    hourglass->to(device);

    int batch_size = data["training"]["batch-size"];
    int kNumberOfEpochs = data["training"]["epochs"];
    const int64_t batches_per_epoch = std::ceil(data_set.size().value() / static_cast<double>(batch_size));

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(data_set),
        batch_size);

    std::vector<float> all_losses;

    torch::optim::Adam optimizer(
        hourglass->parameters(), torch::optim::AdamOptions(2e-4));

    if (data["training"]["load-weights"]["load"])
    {
        try
        {
            torch::load(hourglass,data["training"]["load-weights"]["path"]);
            std::cout << "Weights loaded" << std::endl;
        }
        catch (const c10::Error &e)
        {
            std::cout << e.msg() << std::endl;
            std::cout << "Couldn't load previous weights" << std::endl;
        }
    }

    std::cout << "Beginning Training for " << kNumberOfEpochs << " Epochs" << std::endl;

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
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
                      << epoch << "/" << kNumberOfEpochs << "]"
                      << "[" << ++batch_index << "/" << batches_per_epoch << "]"
                      << " loss: " << loss.item<float>() << std::flush;
        }
    }

    torch::save(hourglass, data["training"]["save-name"]);
    torch::save(optimizer, "hourglass-optimizer.pt");
}
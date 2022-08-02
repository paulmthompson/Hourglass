#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "hourglass.hpp"
#include "dataload.hpp"

#include <vector>
#include <cmath>

using namespace torch;

#pragma once

template <class T>
void train_hourglass(StackedHourglass& hourglass, T& data_set,torch::Device device) {

    hourglass->to(device);

    int batch_size = 8;
    int kNumberOfEpochs = 2;

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(data_set), 
      batch_size);

    std::vector<float> all_losses;

    torch::optim::Adam optimizer(
      hourglass->parameters(), torch::optim::AdamOptions(2e-4));

    const int64_t batches_per_epoch = std::ceil(data_set.size().value() / static_cast<double>(batch_size));

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
      for (auto& batch : *data_loader) {

        hourglass->zero_grad();

        auto data = batch.data.to(device);
        auto labels = batch.target.to(device);

        auto output = hourglass->forward(data);

        std::vector<torch::Tensor> losses;
        for (auto& level_output : output) {
          losses.push_back(torch::mse_loss(level_output, labels));
        }
        torch::Tensor loss = losses[0];
        for (int i = 1 ; i<losses.size(); i ++) {
          //try {
          loss += losses[i];
          //} catch (const c10::Error& e) {
          // std::cout << e.msg() << std::endl;
          //}
        }

        loss.backward();

        optimizer.step();

        all_losses.push_back(loss.item<float>());

        std::cout << "\r"
        "[" << epoch << "/" << kNumberOfEpochs << "]" <<
        "[" << ++batch_index << "/" << batches_per_epoch << "]" <<
        " loss: " << loss.item<float>() << std::flush;
      }
    }

}
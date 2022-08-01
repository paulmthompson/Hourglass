#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "hourglass.hpp"
#include "dataload.hpp"

#include <vector>

using namespace torch;

#pragma once

void train_hourglass(MyDataset& data_set) {

    int batch_size = 8;
    int kNumberOfEpochs = 2;
    int input_dimension = 1;
    int output_dimension = 1;

    const int64_t batches_per_epoch =
      std::ceil(data_set.size().value() / static_cast<double>(batch_size));

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(data_set), 
      batch_size);

    std::vector<float> all_losses;
    
    StackedHourglass hourglass(4,64,input_dimension,output_dimension);

    torch::optim::Adam optimizer(
      hourglass->parameters(), torch::optim::AdamOptions(2e-4));

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
      for (auto& batch : *data_loader) {

        hourglass->zero_grad();

        //auto& data = batch.data;
        //auto& labels = batch.target;

        auto output = hourglass->forward(batch.data);

        std::vector<torch::Tensor> losses;
        for (auto& level_output : output) {
          losses.push_back(torch::mse_loss(level_output, batch.target));
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
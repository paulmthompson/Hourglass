#include <torch/torch.h>
#include "include/hourglass.hpp"
#include "include/dataload.hpp"
//#include "include/training.hpp"

#include <cxxopts.hpp>

#include <iostream>
#include <numeric>
#include <cmath>

int main(int argc, char** argv) {

  cxxopts::Options options("Hourglass Neural Network", "Commandline interface for hourglass neural network");

  options.add_options()
    ("h,help", "Print usage")
    ("d,data", "Data JSON configuration file", cxxopts::value<std::string>())
  ;

  auto result = options.parse(argc, argv);

  if (result.count("help"))
  {
    std::cout << options.help() << std::endl;
    exit(0);
  } else if (result.count("data")) {

    std::cout << "Configuration file is " << result["data"].as<std::string>() << std::endl;

    auto data_set = MyDataset(result["data"].as<std::string>()).map(torch::data::transforms::Stack<>());

    int batch_size = 8;
    int kNumberOfEpochs = 2;

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(data_set), 
      batch_size);
    
    StackedHourglass hourglass = createHourglass(result["data"].as<std::string>());

    torch::optim::Adam optimizer(
      hourglass->parameters(), torch::optim::AdamOptions(2e-4));

    std::vector<float> all_losses;
    const int64_t batches_per_epoch = std::ceil(data_set.size().value() / static_cast<double>(batch_size));

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
      for (auto& batch : *data_loader) {

        hourglass->zero_grad();

        auto& data = batch.data;
        auto& labels = batch.target;

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

    exit(0);
  }

  return 0;
}
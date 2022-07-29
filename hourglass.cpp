#include <torch/torch.h>
#include "include/hourglass.hpp"
#include "include/dataload.hpp"

#include <cxxopts.hpp>

#include <iostream>
#include <numeric>

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
    std::cout << "entering data method" << std::endl;

    std::cout << "Configuration file is " << result["data"].as<std::string>() << std::endl;

    auto data_set = MyDataset(result["data"].as<std::string>()).map(torch::data::transforms::Stack<>());

    std::cout << "Data Loaded" << std::endl;
    std::cout << "Data size is " << data_set.size().value() << std::endl;

    int batch_size = 8;
    int kNumberOfEpochs = 2;
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(data_set), 
      batch_size);

    std::vector<float> all_losses;
    
    StackedHourglass hourglass(4,64,1,1);

    torch::optim::Adam optimizer(
      hourglass->parameters(), torch::optim::AdamOptions(2e-4));

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
      for (torch::data::Example<>& batch : *data_loader) {

        hourglass->zero_grad();

        auto data = batch.data;
        auto labels = batch.target;

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

        std::printf(
            "\r[%2ld/%2ld][%3ld] D_loss: %.4f",
            epoch,
            kNumberOfEpochs,
            ++batch_index,
            loss.item<float>());
      }
    }

    exit(0);
  }

  return 0;
}
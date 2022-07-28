#include <torch/torch.h>
#include "include/hourglass.hpp"
#include "include/dataload.hpp"

#include <cxxopts.hpp>

#include <iostream>

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

        std::cout << "Hourglass evaluated" << std::endl;

        std::vector<torch::Tensor> losses;
        for (auto& level_output : output) {
          losses.push_back(torch::mse_loss(level_output, labels));
        }
        for (auto& loss : losses) {
          loss.backward();
        }

        optimizer.step();
        std::printf(
            "\r[%2ld/%2ld][%3ld] D_loss: %.4f",
            epoch,
            kNumberOfEpochs,
            ++batch_index,
            losses[losses.size()].item<float>());
      }
    }


    exit(0);
  }


  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;


  return 0;
}
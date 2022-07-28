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

    int batch_size = 16;
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(data_set), 
      batch_size);
    
    for (torch::data::Example<>& batch : *data_loader) {

      auto data = batch.data;
      auto labels = batch.target;

      std::cout << "Batch size: " << data.size(0) << std::endl;
    }


    exit(0);
  }


  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;


  return 0;
}
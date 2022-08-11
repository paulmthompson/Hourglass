#include <torch/torch.h>
#include "include/hourglass.hpp"
#include "include/dataload.hpp"
#include "include/training.hpp"
#include "include/prediction.hpp"

#include <cxxopts.hpp>

#include <iostream>
#include <numeric>
#include <cmath>

int main(int argc, char** argv) {

  cxxopts::Options options("Hourglass Neural Network", "Commandline interface for hourglass neural network");

  options.add_options()
    ("h,help", "Print usage")
    ("d,data", "Data JSON configuration file", cxxopts::value<std::string>())
    ("t,train", "Perform training on dataset", cxxopts::value<bool>()->default_value("true"))
    ("p,predict","Predict using trained network",cxxopts::value<bool>()->default_value("true"))
  ;

  auto result = options.parse(argc, argv);

  if (result.count("help"))
  {
    std::cout << options.help() << std::endl;
    exit(0);
  } else if (result.count("data")) {

    std::cout << "Configuration file is " << result["data"].as<std::string>() << std::endl;

    auto data_set = MyDataset(result["data"].as<std::string>()).map(torch::data::transforms::Stack<>());

    StackedHourglass hourglass = createHourglass(result["data"].as<std::string>());

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! Using the GPU." << std::endl;
      device = torch::Device(torch::kCUDA);
    }


    if (result["train"].as<bool>()) {
      train_hourglass(hourglass,data_set,device,result["data"].as<std::string>());
    }

    if (result["predict"].as<bool>()) {
      predict(hourglass,data_set,device,result["data"].as<std::string>());
    }

    exit(0);
  }

  return 0;
}
#include <torch/torch.h>
#include "include/hourglass.hpp"
#include "include/dataload.hpp"
#include "include/training.hpp"

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

    StackedHourglass hourglass = createHourglass(result["data"].as<std::string>());

    train_hourglass(hourglass,data_set);

    exit(0);
  }

  return 0;
}
#include <torch/torch.h>
#include "include/hourglass.hpp"
#include "dataloader/dataload.hpp"
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

    std::string config_file = result["data"].as<std::string>();
    
    std::cout << "Configuration file is " << config_file << std::endl;
    
    torch::Device device(torch::kCPU);

    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! Using the GPU." << std::endl;
      device = torch::Device(torch::kCUDA);
    }
    
    StackedHourglass hourglass = createHourglass(config_file);
    hourglass->to(kFloat32);
    
    if (result["train"].as<bool>()) {

      auto training_opts = training_options(config_file);

      auto data_set = MyDataset(training_opts).map(torch::data::transforms::Stack<>());
      train_hourglass(hourglass,data_set,device,training_opts);

      training_opts.image_augmentation = false;
      data_set = MyDataset(training_opts).map(torch::data::transforms::Stack<>());
      
      //Recreate hourglass?
      hourglass = createHourglass(config_file);
      load_weights(hourglass,training_opts.weight_save_name);

      predict(hourglass,data_set,device,config_file);
    }

    if (result["predict"].as<bool>()) {
      //auto data_set = MyDataset(config_file).map(torch::data::transforms::Stack<>());
      //predict(hourglass,data_set,device,config_file);
      predict_video(hourglass,device,config_file);
    }
    
    exit(0);
  }

  return 0;
}
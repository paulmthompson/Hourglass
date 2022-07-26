#include <torch/torch.h>
#include "include/hourglass.hpp"
#include "include/dataload.hpp"

#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
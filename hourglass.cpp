#include <torch/torch.h>
#include "include/residual.hpp"
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
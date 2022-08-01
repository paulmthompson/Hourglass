 #include <torch/torch.h>

 #include <iostream>

 using namespace torch;

 #pragma once

struct Conv0Impl : nn::Module {
    Conv0Impl(int input_dim, int output_dim, int kernel_size,int stride, int padding)
    : conv1(nn::Conv2dOptions(input_dim,output_dim,kernel_size)
            .stride(stride)
            .padding(padding))
{
    register_module("conv1",conv1);
}
 torch::Tensor forward(torch::Tensor x) {
   x = conv1(x);
   return x;
 }
 nn::Conv2d conv1;
};
TORCH_MODULE(Conv0);

struct ResidualImpl : nn::Module {
    ResidualImpl(int input_dim, int output_dim)
        : batch_norm1(input_dim),
        conv1(nn::Conv2dOptions(input_dim,output_dim / 2,1)
            .stride(1)
            .padding(0)),
        batch_norm2(output_dim / 2),
        conv2(nn::Conv2dOptions(output_dim / 2, output_dim / 2, 3)
            .stride(1)
            .padding(1)),
        batch_norm3(output_dim / 2),
        conv3(nn::Conv2dOptions(output_dim / 2, output_dim, 1)
            .stride(1)
            .padding(0))
 {
   register_module("conv1", conv1);
   register_module("conv2", conv2);
   register_module("conv3", conv3);
   register_module("batch_norm1", batch_norm1);
   register_module("batch_norm2", batch_norm2);
   register_module("batch_norm3", batch_norm3);
 }

   torch::Tensor forward(torch::Tensor input) {
   torch::Tensor x = conv1(torch::relu(batch_norm1(input)));
   x = conv2(torch::relu(batch_norm2(x)));
   x = conv3(torch::relu(batch_norm3(x)));
   return x + input;
 }
    nn::Conv2d conv1, conv2, conv3;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(Residual);


struct Residual_SkipImpl : nn::Module {
    Residual_SkipImpl(int input_dim, int output_dim)
        : conv1(nn::Conv2dOptions(input_dim,output_dim,1)
            .stride(1)
            .padding(0)),
        batch_norm1(input_dim),
        conv2(nn::Conv2dOptions(input_dim,output_dim / 2,1)
            .stride(1)
            .padding(0)),
        batch_norm2(output_dim / 2),
        conv3(nn::Conv2dOptions(output_dim / 2, output_dim / 2, 3)
            .stride(1)
            .padding(1)),
        batch_norm3(output_dim / 2),
        conv4(nn::Conv2dOptions(output_dim / 2, output_dim, 1)
            .stride(1)
            .padding(0))
 {
   register_module("conv1", conv1);
   register_module("conv2", conv2);
   register_module("conv3", conv3);
   register_module("conv4", conv4);
   register_module("batch_norm1", batch_norm1);
   register_module("batch_norm2", batch_norm2);
   register_module("batch_norm3", batch_norm3);
 }

  torch::Tensor forward(torch::Tensor input) {
   torch::Tensor x = conv2(torch::relu(batch_norm1(input)));
   x = conv3(torch::relu(batch_norm2(x)));
   x = conv4(torch::relu(batch_norm3(x)));
   return x + conv1(input);
 }
    nn::Conv2d conv1, conv2, conv3, conv4;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(Residual_Skip);


 #include <torch/torch.h>

 #include "residual.hpp"

 using namespace torch;

struct FirstLayerImpl : nn::Module {
    FirstLayerImpl(int N)
        : batch_norm1(3),
        conv1(nn::ConvTranspose2dOptions(3,64,7)
            .stride(2)
            .padding(3)),
        r1(64,128),
        p1(nn::MaxPool2dOptions({2,2})),
        r2(128,128),
        r3(128,N)
 {
   register_module("conv1", conv1);
   register_module("batch_norm1",batch_norm1);
   register_module("r1",r1);
   register_module("p1",p1);
   register_module("r2",r2);
   register_module("r3",r3);
 }

  torch::Tensor forward(torch::Tensor input) {
   torch::Tensor x = conv1(torch::relu(batch_norm1(input)));
   x = r1(x);
   x = p1(x);
   x = r2(x);
   x = r3(x);
   return x;
 }
    nn::ConvTranspose2d conv1;
    nn::BatchNorm2d batch_norm1;
    Residual_Skip r1,r3;
    nn::MaxPool2d p1;
    Residual r2;
};
TORCH_MODULE(FirstLayer);
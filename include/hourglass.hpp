
 #include <torch/torch.h>
 #include <nlohmann/json.hpp>

 #include "residual.hpp"

 #include <vector>
 #include <string>
 #include <iostream>

 using namespace torch;
 using json = nlohmann::json;

 #pragma once

struct FirstLayerImpl : nn::Module {
    FirstLayerImpl(int N_Channel, int N_Input_Channel)
        : batch_norm1(N_Input_Channel),
        conv1(nn::Conv2dOptions(N_Input_Channel,64,7)
            .stride(2)
            .padding(3)),
        r1(64,128),
        p1(nn::MaxPool2dOptions({2,2})),
        r2(128,128),
        r3(128,N_Channel)
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
    nn::Conv2d conv1;
    nn::BatchNorm2d batch_norm1;
    Residual_Skip r1,r3;
    nn::MaxPool2d p1;
    Residual r2;
};
TORCH_MODULE(FirstLayer);

struct OutLayerImpl : nn::Module {
    OutLayerImpl(int N)
        : r1(N,N),
        batch_norm1(N),
        conv1(nn::Conv2dOptions(N,N,1))
 {
   register_module("r1",r1);
   register_module("batch_norm1",batch_norm1);
   register_module("conv1",conv1);
 }

  torch::Tensor forward(torch::Tensor input) {
   torch::Tensor x = r1(input);
   x = conv1(x);
   x = torch::relu(batch_norm1(x));
   return x;
 }
    nn::Conv2d conv1;
    nn::BatchNorm2d batch_norm1;
    Residual r1;
};
TORCH_MODULE(OutLayer);

struct HourglassImpl : nn::Module {
    HourglassImpl(int N)
        : branch_res1(N,N),
        p1(nn::MaxPool2dOptions({2,2})),
        res1(N,N),

        branch_res2(N,N),
        p2(nn::MaxPool2dOptions({2,2})),
        res2(N,N),

        branch_res3(N,N),
        p3(nn::MaxPool2dOptions({2,2})),
        res3(N,N),

        branch_res4(N,N),
        p4(nn::MaxPool2dOptions({2,2})),
        res4(N,N),

        res5(N,N),

        res6(N,N),
        unpool4(nn::MaxUnpool2dOptions({2,2})),

        res7(N,N),
        unpool3(nn::MaxUnpool2dOptions({2,2})),

        res8(N,N),
        unpool2(nn::MaxUnpool2dOptions({2,2})),

        res9(N,N),
        unpool1(nn::MaxUnpool2dOptions({2,2}))
        
 {
   register_module("branch_res1",branch_res1);
   register_module("p1",p1); // 128 x 128 -> 64 x 64
   register_module("res1",res1);

   register_module("branch_res2",branch_res2);
   register_module("p2",p2); // 64 x 64 -> 32 x 32
   register_module("res2",res2);

   register_module("branch_res3",branch_res3);
   register_module("p3",p3); // 32 x 32 -> 16 x 16
   register_module("res3",res3);

   register_module("branch_res4",branch_res4);
   register_module("p4",p4); // 16 x 16 -> 8 x 8
   register_module("res4",res4);

   register_module("res5",res5);

   register_module("res6",res6);
   register_module("unpool4",unpool4); // 8 x 8 -> 16 x 16

   register_module("res7",res7);
   register_module("unpool3",unpool3); // 16 x 16 -> 32 x 32

   register_module("res8",res8);
   register_module("unpool2",unpool2); // 32 x 32 -> 64 x 64

   register_module("res9",res9);
   register_module("unpool1",unpool1); // 64 x 64 -> 128 x 128

 }

  torch::Tensor forward(torch::Tensor input) {

   torch::Tensor up1 = branch_res1(input);
   auto [out1, indices1]= p1->forward_with_indices(input);
   out1 = res1(out1);

   torch::Tensor up2 = branch_res2(out1);
   auto  [out2,indices2] = p2->forward_with_indices(out1);
   out2 = res2(out2);

   torch::Tensor up3 = branch_res3(out2);
   auto [out3, indices3] = p3->forward_with_indices(out2);
   out3 = res3(out3);

   torch::Tensor up4 = branch_res4(out3);
   auto [out4, indices4] = p4->forward_with_indices(out3);
   out4 = res4(out4);

   out4 = res5(out4);

   out4 = res6(out4);
   torch::Tensor x = unpool4(out4,indices4) + up4;

   x = res7(x);
   x = unpool3(x,indices3) + up3;
  
   x = res8(x);
   x = unpool2(x,indices2) + up2;

   x = res9(x);
   x = unpool1(x,indices1) + up1;

   return x;
 }
    Residual branch_res1, branch_res2, branch_res3, branch_res4;
    Residual res1, res2, res3, res4, res5, res6,res7, res8, res9;
    nn::MaxPool2d p1, p2, p3, p4;
    nn::MaxUnpool2d unpool4,unpool3,unpool2,unpool1;
}; 
TORCH_MODULE(Hourglass);

struct StackedHourglassImpl : nn::Module {
    StackedHourglassImpl(int nstack,int N,int k,int N_Input_Channel)
        : fb(N,N_Input_Channel),
        hg(nstack,Hourglass(N)),
        o1(nstack,OutLayer(N)),
        c1(nstack,Conv0(N,k,1,1,0)),
        merge_features(nstack-1,Conv0(N,N,1,1,0)),
        merge_preds(nstack-1,Conv0(k,N,1,1,0)),
        input_dims(N_Input_Channel),
        output_dims(k)

 {
   register_module("fb",fb);
   for (int i=0; i<nstack;i++) {

    std::string hg_tmp_name = "hg" + std::to_string(i);
    std::string o1_tmp_name = "o1" + std::to_string(i);
    std::string c1_tmp_name = "c1" + std::to_string(i);

    register_module(hg_tmp_name,hg[i]);
    register_module(o1_tmp_name,o1[i]);
    register_module(c1_tmp_name,c1[i]);

   }
   for (int i=0; i<nstack-1;i++) {

    std::string features_tmp_name = "features" + std::to_string(i);
    std::string preds_tmp_name = "preds" + std::to_string(i);

    register_module(features_tmp_name,merge_features[i]);
    register_module(preds_tmp_name,merge_preds[i]);

   }
 }

  std::vector<torch::Tensor> forward(torch::Tensor input) {

   torch::Tensor x = fb(input);

   std::vector<torch::Tensor> temps;
   std::vector<torch::Tensor> preds;

   temps.push_back(x);

   for (int i=0; i<hg.size(); i++) {
    torch::Tensor hg_out = hg[i](temps[i]);

    torch::Tensor features = o1[i](hg_out);

    torch::Tensor pred = c1[i](features);

    preds.push_back(pred);
    if (i < hg.size()-1) {
      torch::Tensor m_features = merge_features[i](features);

      torch::Tensor m_preds = merge_preds[i](pred);

      torch::Tensor temp1 = m_features + m_preds;

      temps.push_back(temp1 + temps[i]);
    }
   }
   
   return preds;
 }

    int get_input_dims() const {return input_dims;}
    int get_output_dims() const {return output_dims;}

    FirstLayer fb;
    std::vector<Hourglass> hg;
    std::vector<OutLayer> o1;
    std::vector<Conv0> c1;
    std::vector<Conv0> merge_features;
    std::vector<Conv0> merge_preds;
    int input_dims;
    int output_dims;
};
TORCH_MODULE(StackedHourglass);

StackedHourglass createHourglass(const std::string& config_file) {

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    const int stacks = data["hourglass"]["stacks"];
    const int channels = data["hourglass"]["channels"];
    const int input_dimensions = data["hourglass"]["input-dimensions"];
    const int output_dimensions = data["hourglass"]["output-dimensions"];

    StackedHourglass hourglass(stacks,channels,output_dimensions,input_dimensions);

    return hourglass;
}

void load_weights(StackedHourglass& hourglass, const std::string& weight_name) {

    std::filesystem::path weight_path = weight_name;
    if (std::filesystem::exists(weight_path)) {
        try
        {
            torch::load(hourglass,weight_path);
            std::cout << "Weights loaded" << std::endl;
        }
        catch (const c10::Error &e)
        {
            std::cout << e.msg() << std::endl;
            std::cout << "Couldn't load previous weights" << std::endl;
        }
    } else {
    std::cout << "Could not load previous weights at path: " << weight_path.string() << std::endl;
    }

}
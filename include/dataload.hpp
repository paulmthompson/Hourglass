 #include <torch/torch.h>
 #include <nlohmann/json.hpp>

 #include <iostream>

 using namespace torch;
 using json = nlohmann::json;


void read_json_file(const std::string& config_file) {

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    std::cout << data["name"] << std::endl;
};


 //https://discuss.pytorch.org/t/libtorch-how-to-use-torch-datasets-for-custom-dataset/34221/2
 torch::Tensor read_img(const std::string& config_file) 
 {
    torch::Tensor tensor;
    return tensor;
 };

 torch::Tensor read_labels(const std::string& config_file)
 {
    torch::Tensor tensor;
    return tensor;
 };

 class MyDataset : public torch::data::Dataset<MyDataset>
{
    private:
        torch::Tensor states_, labels_;

    public:
        explicit MyDataset(const std::string& config_file) 
            : states_(read_img(config_file)),
              labels_(read_labels(config_file))
        {

        };
        torch::data::Example<> get(size_t index) override;
};

torch::data::Example<> MyDataset::get(size_t index)
{
    // You may for example also read in a .csv file that stores locations
    // to your data and then read in the data at this step. Be creative.
    return {states_[index], labels_[index]};
} 
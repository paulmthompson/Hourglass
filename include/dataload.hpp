#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <string>
#include <iostream>
#include <filesystem>
#include <optional>

using namespace torch;
namespace fs = std::filesystem;
using json = nlohmann::json;

//https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
std::optional<std::string> match_folder_in_path(const std::filesystem::path& dir_path,std::string folder_path) {

    for (const auto & entry : fs::directory_iterator(dir_path)) {
        if (entry.path().string().find(folder_path) != string::npos) {
            return entry.path().string();
        }
    }
    return std::nullopt;
};

std::filesystem::path generate_output_path_from_json(std::filesystem::path folder_path, const json& subpaths_json) {

    for (const auto& sub_path : subpaths_json) {
        std::optional<std::string> matched_folder = match_folder_in_path(folder_path,sub_path);
        if (matched_folder) {
            folder_path /= matched_folder.value();
        }
    }
    return folder_path;

};

void add_image_to_load(std::vector<std::string>& img_files, const std::filesystem::path& folder_path, const json& json_filetypes) {
    
    for (const auto & file_type : json_filetypes) {
        for (const auto & entry : fs::directory_iterator(folder_path)) {
            if (entry.path().extension() == file_type) {
                std::filesystem::path image_file_path = folder_path / entry.path();
                img_files.push_back(image_file_path.string());
            }
        }
    }

};

void read_json_file(const std::string& config_file) {

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    std::cout << "Configuration file found with name " << data["name"] << std::endl;

    std::filesystem::path data_path = data["folder_path"];

    std::vector<std::string> img_files;
    std::vector<std::string> label_files;

    for (const auto& entry : data["experiments"]) {

        std::filesystem::path experiment_path = data_path / entry["name"];

        std::cout << "Loading experiment path: " << experiment_path << std::endl;

        for (const auto& this_view : data["images"]["views"]) {

            std::filesystem::path img_folder_path = experiment_path;
            
            img_folder_path = generate_output_path_from_json(img_folder_path, this_view["prefix"]);

            std::cout << "This experiment image data is " << img_folder_path << std::endl;

            add_image_to_load(img_files,img_folder_path,data["images"]["filetypes"]);

            std::filesystem::path label_folder_path = experiment_path;

            label_folder_path = generate_output_path_from_json(label_folder_path, this_view["label_prefix"]);

            std::cout << "This experiment label data is " << label_folder_path << std::endl;

            add_image_to_load(label_files,label_folder_path,data["labels"]["filetypes"]);
        }  
    }

    std::cout << "The total number of images is " << img_files.size() << std::endl;
    std::cout << "The total number of labels is " << label_files.size() << std::endl;
    
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
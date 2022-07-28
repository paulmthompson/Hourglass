#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <filesystem>
#include <optional>
#include <regex>
#include <tuple>

using namespace torch;
namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace cv;

using paths = std::vector<std::filesystem::path>;

//https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
std::optional<std::string> match_folder_in_path(const std::filesystem::path& dir_path,std::string folder_path) {

    for (const auto & entry : fs::directory_iterator(dir_path)) {
        if (entry.path().string().find(folder_path) != string::npos) {
            return entry.path().string();
        }
    }
    return std::nullopt;
};

std::filesystem::path generate_output_path_from_json(const std::filesystem::path& folder_path, const json& subpaths_json) {

    std::filesystem::path output_path = folder_path;

    for (const auto& sub_path : subpaths_json) {
        std::optional<std::string> matched_folder = match_folder_in_path(output_path,sub_path);
        if (matched_folder) {
            output_path /= matched_folder.value();
        }
    }
    return output_path;
};

struct name_and_path {
    std::string name;
    std::filesystem::path path;
};

std::vector<name_and_path> add_image_to_load(const std::filesystem::path& folder_path, const json& json_filetypes, const json& json_prefix) {
    
    std::vector<name_and_path> out_images;

    std::regex image_regex(json_prefix);
    for (const auto & file_type : json_filetypes) {
        for (const auto & entry : fs::directory_iterator(folder_path)) {
            if (entry.path().extension() == file_type) {
                std::filesystem::path image_file_path = folder_path / entry.path();
                std::string image_name = std::regex_replace(entry.path().stem().string(), image_regex, "");
                out_images.push_back(name_and_path{image_name,image_file_path});
            }
        }
    }
    return out_images;
};

//https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-3-advanced-libtorch/
torch::Tensor load_image(const std::filesystem::path& image_path) {
    cv::Mat image = cv::imread(image_path.string(),cv::IMREAD_UNCHANGED);

    if (!image.isContinuous()) {   
        image = image.clone(); 
    }

    // (We use torch::empty here since it can be somewhat faster than `zeros` //  or `ones` since it is allowed to fill the tensor with garbage.) 
    auto tensor = torch::empty(
           { image.rows, image.cols, image.channels()},
            // Set dtype=byte and place on CPU, you can change these to whatever   
            // suits your use-case.   
            torch::TensorOptions()  
               .dtype(torch::kByte)   
               .device(torch::kCPU));     
               
    // Copy over the data 
    std::memcpy(tensor.data_ptr(), reinterpret_cast<void*>(image.data), tensor.numel() * sizeof(at::kByte));

    return tensor;
};

 //https://discuss.pytorch.org/t/libtorch-how-to-use-torch-datasets-for-custom-dataset/34221/2
 torch::Tensor read_images(const std::vector<std::filesystem::path>& image_paths) 
 {
    std::vector<torch::Tensor> tensor;

    for (auto& img_file : image_paths) {
        tensor.push_back(load_image(img_file));
    }

    auto stacked = torch::stack(torch::TensorList(tensor));

    std::cout << "The size of image array is " << stacked.sizes() << std::endl;

    return stacked;
 };

std::tuple<paths,paths> read_json_file(const std::string& config_file) {

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    std::cout << "Configuration file found with name " << data["name"] << std::endl;

    std::filesystem::path data_path = data["folder_path"];

    std::vector<std::filesystem::path> img_files;
    std::vector<std::filesystem::path> label_files; // This should be vector of vectors to account for different label sizes

    for (const auto& entry : data["experiments"]) {

        auto experiment_path = data_path / entry["name"];

        std::cout << "Loading experiment path: " << experiment_path << std::endl;

        for (const auto& this_view : data["images"]["views"]) {

            auto img_folder_path = generate_output_path_from_json(experiment_path, this_view["prefix"]);

            std::cout << "This experiment image data is " << img_folder_path << std::endl;

            std::vector<name_and_path> this_view_images = add_image_to_load(img_folder_path,data["images"]["filetypes"],data["images"]["name_prefix"]);

            auto label_folder_path = generate_output_path_from_json(experiment_path, this_view["label_prefix"]);

            std::cout << "This experiment label data is " << label_folder_path << std::endl;

            std::vector<name_and_path> this_view_labels = add_image_to_load(label_folder_path,data["labels"]["filetypes"],data["labels"]["name_prefix"]);

            for (const auto& this_img : this_view_images) {
                for (const auto& this_label : this_view_labels) {
                    if (this_img.name.compare(this_label.name) == 0) {
                        img_files.push_back(this_img.path);
                        label_files.push_back(this_label.path);
                        break;
                    }
                }
            }

        }  
    }

    std::cout << "The total number of images is " << img_files.size() << std::endl;
    std::cout << "The total number of labels is " << label_files.size() << std::endl;

    return make_tuple(img_files,label_files);
};

 class MyDataset : public torch::data::Dataset<MyDataset>
{
    private:
        torch::Tensor states_, labels_;

    public:
        explicit MyDataset(const std::string& config_file) 
        {
            auto [img_files, label_files] = read_json_file(config_file);

            states_ = read_images(img_files);
            std::cout << "Images Loaded" << std::endl;

            labels_ = read_images(label_files);
            std::cout << "Labels Loaded" << std::endl;
        };
        torch::data::Example<> get(size_t index) override;
        optional<size_t> size() const override;
};

torch::data::Example<> MyDataset::get(size_t index)
{
    // You may for example also read in a .csv file that stores locations
    // to your data and then read in the data at this step. Be creative.
    return {states_[index], labels_[index]};
};

optional<size_t> MyDataset::size() const { return states_.size(0); }
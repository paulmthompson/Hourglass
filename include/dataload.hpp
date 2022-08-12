#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <ffmpeg_wrapper/videodecoder.h>

#include <string>
#include <iostream>
#include <filesystem>
#include <optional>
#include <regex>
#include <tuple>
#include <algorithm>
#include <vector>

#include "augmentation.hpp"

using namespace torch;
namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace cv;

using paths = std::vector<std::filesystem::path>;

#pragma once

struct img_label_pair {
    img_label_pair() {}
    img_label_pair(fs::path this_img, fs::path this_label) {
        img = this_img;
        labels.push_back(this_label);
    }
    fs::path img;
    paths labels;
};

torch::Tensor make_tensor_stack(std::vector<torch::Tensor>& tensor) {
    auto stacked = torch::stack(torch::TensorList(tensor));

    return stacked.to(torch::kFloat32).div(255);
}

torch::Tensor LoadFrame(ffmpeg_wrapper::VideoDecoder& vd, int frame_id) {

    std::vector<uint8_t> image = vd.getFrame(frame_id, true);

    int img_height = vd.getHeight();
    int img_width = vd.getWidth();

    auto tensor = torch::empty(
           { img_height, img_width, 1},
            torch::TensorOptions()  
               .dtype(torch::kByte)   
               .device(torch::kCPU));     
               
    // Copy over the data 
    std::memcpy(tensor.data_ptr(), image.data(), tensor.numel() * sizeof(at::kByte));

    return tensor.permute({2,0,1});
}

torch::Tensor LoadFrames(ffmpeg_wrapper::VideoDecoder& vd, int frame_start, int frame_end) {

    std::vector<torch::Tensor> frames;

    for (int i = frame_start; i <= frame_end; i++) {
        frames.push_back(LoadFrame(vd,i));
    }

    //std::cout << "Loaded frames " << frame_start << " - " << frame_end << std::endl;

    return make_tensor_stack(frames);
}

template<typename T>
void shuffle(std::vector<T>& imgs, std::vector<T>& labels) {

    int n = imgs.size();
    //https://www.techiedelight.com/shuffle-vector-cpp/
    for (int i = 0; i < n - 1; i++)
    {
        int j = i + rand() % (n - i);
        std::swap(imgs[i],imgs[j]);
        std::swap(labels[i],labels[j]);
    }
};

//https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
std::optional<std::string> match_folder_in_path(const fs::path& dir_path,std::string folder_path) {

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
            if (entry.path().extension() == file_type.get<std::string>()) {
                std::filesystem::path image_file_path = folder_path / entry.path();
                std::string image_name = std::regex_replace(entry.path().stem().string(), image_regex, "");
                out_images.push_back(name_and_path{image_name,image_file_path});
            }
        }
    }
    return out_images;
};

torch::Tensor convert_to_tensor(cv::Mat& image) {

    auto tensor = torch::empty(
           { image.rows, image.cols, image.channels()},
            torch::TensorOptions()  
               .dtype(torch::kByte)   
               .device(torch::kCPU));     
               
    // Copy over the data 
    std::memcpy(tensor.data_ptr(), reinterpret_cast<void*>(image.data), tensor.numel() * sizeof(at::kByte));

    return tensor.permute({2,0,1});
}

//https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-3-advanced-libtorch/
cv::Mat load_image(const std::filesystem::path& image_path, int w, int h) {
    cv::Mat raw_image = cv::imread(image_path.string(),cv::IMREAD_GRAYSCALE);
    cv::Mat image;
    cv::resize(raw_image, image,cv::Size(w,h), cv::INTER_AREA);

    if (!image.isContinuous()) {   
        image = image.clone(); 
    }

    return image;
};

std::tuple<int,int> get_width_height(const std::string& config_file, const std::string& keyword) {

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    std::vector<int> height_width = data[keyword]["resolution"];

    return std::make_tuple(height_width[0],height_width[1]);
};

 //https://discuss.pytorch.org/t/libtorch-how-to-use-torch-datasets-for-custom-dataset/34221/2
 std::tuple<torch::Tensor,torch::Tensor> read_images(const std::vector<img_label_pair> image_paths, const std::string& config_file)
 {
    auto [w_img, h_img] = get_width_height(config_file,"images");
    auto [w_label, h_label] = get_width_height(config_file,"labels");

    std::vector<torch::Tensor> img_tensor;
    std::vector<torch::Tensor> label_tensor;

    for (auto& this_img_label : image_paths) {
        auto this_img = load_image(this_img_label.img,w_img,h_img);
        auto this_label = load_image(this_img_label.labels[0],w_label,h_label);

        auto [aug_img, aug_label]  = image_augmentation(this_img,this_label);

        for (int i = 0; i < aug_img.size(); i++) {
            img_tensor.push_back(convert_to_tensor(aug_img[i]));
            label_tensor.push_back(convert_to_tensor(aug_label[i]));
        }
    }

    shuffle(img_tensor,label_tensor);

    return std::make_tuple(make_tensor_stack(img_tensor),make_tensor_stack(label_tensor));
 };

std::vector<img_label_pair> read_json_file(const std::string& config_file) {

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    std::cout << "Configuration file found with name " << data["name"] << std::endl;

    std::filesystem::path data_path = data["folder_path"];

    std::vector<img_label_pair> img_label_files;

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
                        img_label_files.push_back(img_label_pair(this_img.path,this_label.path));
                        break;
                    }
                }
            }

        }  
    }

    std::cout << "The total number of images is " << img_label_files.size() << std::endl;

    return img_label_files;
};



 class MyDataset : public torch::data::Dataset<MyDataset>
{
    private:
        torch::Tensor states_, labels_;

    public:
        explicit MyDataset(const std::string& config_file) 
        {
            auto img_label_files = read_json_file(config_file);

            auto [states, labels] = read_images(img_label_files, config_file);

            states_ = std::move(states);
            labels_ = std::move(labels);

            std::cout << "Image size is " << states_.size(0) << std::endl;
            std::cout << "Label size is " << labels_.size(0) << std::endl;
        };
        torch::data::Example<> get(size_t index) override;
        torch::optional<size_t> size() const override;
};

torch::data::Example<> MyDataset::get(size_t index)
{
    // You may for example also read in a .csv file that stores locations
    // to your data and then read in the data at this step. Be creative.
    return {states_[index], labels_[index]};
};

torch::optional<size_t> MyDataset::size() const { return states_.size(0); }
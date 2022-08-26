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
#include <memory>

#include "augmentation.hpp"

using namespace torch;
namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace cv;

using paths = std::vector<std::filesystem::path>;

#pragma once

//https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-3-advanced-libtorch/
cv::Mat load_image_from_path(const std::filesystem::path& image_path, int w, int h) {
    cv::Mat raw_image = cv::imread(image_path.string(),cv::IMREAD_GRAYSCALE);
    cv::Mat image;
    cv::resize(raw_image, image,cv::Size(w,h), cv::INTER_AREA);

    if (!image.isContinuous()) {   
        image = image.clone(); 
    }

    return image;
};

/////////////////////////////////////////////////////////////////////////////////
class label_path {
public:
virtual cv::Mat load_image(int w, int h) const = 0;
};

class img_label_path : public label_path {
    public:
    std::filesystem::path path;

    img_label_path(std::filesystem::path this_path) {
        this->path = this_path;
    }

    cv::Mat load_image(int w, int h) const override {
        return load_image_from_path(this->path,w,h);
    }

};

class img_label_pair {
    public:
    img_label_pair() = default;
    img_label_pair(fs::path this_img, fs::path this_label) {
        this->img = this_img;
        this->labels = std::vector<std::unique_ptr<img_label_path>>();
        this->labels.push_back(std::make_unique<img_label_path>(this_label));
    }
    fs::path img;
    std::vector<std::unique_ptr<img_label_path>> labels;
};

/////////////////////////////////////////////////////////////////////////////////

class training_options {
public:
    training_options(const std::string& config_file) {
        std::ifstream f(config_file);
        json data = json::parse(f);
        f.close();

        this->epochs = 1;
        if (data["training"].contains("epochs")) {
            this->epochs = data["training"]["epochs"];
        }
        this->batch_size = 32;
        if (data["training"].contains("batch-size")) {
            this->batch_size = data["training"]["batch-size"];
        }
        this->learning_rate = 5e-5;
        if (data["training"].contains("learning-rate")) {
            this->learning_rate = data["training"]["learning-rate"];
        }
        this->image_augmentation = false;
        if (data["training"].contains("image-augmentation")) {
            this->image_augmentation = data["training"]["image-augmentation"];
        }

        this->config_file = config_file;
    }

    int batch_size;
    int epochs;
    float learning_rate;
    bool image_augmentation;
    std::string config_file;
private:
    
};

/////////////////////////////////////////////////////////////////////////////////

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

typedef enum {MASK, PIXEL} LABEL_TYPE;

struct name_and_path {
    std::string name;
    std::filesystem::path path;
    int x;
    int y;
    LABEL_TYPE label_type;
    name_and_path(std::string name, std::filesystem::path path) {
        this->name = name;
        this->path = path;
        this->x = 0;
        this->y = 0;
        this->label_type = MASK;
    }
    name_and_path(std::filesystem::path path, int x, int y) {
        this->name = "";
        this->path = path;
        this->x = x;
        this->y = y;
        this->label_type = PIXEL;
    }
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

std::tuple<int,int> get_width_height(const std::string& config_file, const std::string& keyword) {

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    std::vector<int> height_width = data[keyword]["resolution"];

    return std::make_tuple(height_width[0],height_width[1]);
};

 //https://discuss.pytorch.org/t/libtorch-how-to-use-torch-datasets-for-custom-dataset/34221/2
 std::tuple<torch::Tensor,torch::Tensor> read_images(const std::vector<img_label_pair>& image_paths, training_options& training_opts)
 {
    auto [w_img, h_img] = get_width_height(training_opts.config_file,"images");
    auto [w_label, h_label] = get_width_height(training_opts.config_file,"labels");

    std::vector<torch::Tensor> img_tensor;
    std::vector<torch::Tensor> label_tensor;

    for (auto& this_img_label : image_paths) {
        auto this_img = load_image_from_path(this_img_label.img,w_img,h_img);

        cv::Mat this_label;
        std::vector<cv::Mat> array_of_labels;
        for (int i=0; i<this_img_label.labels.size(); i++) {
            array_of_labels.push_back(this_img_label.labels[i]->load_image(w_label,h_label));
            //array_of_labels.push_back(load_image_from_path(this_img_label.labels[i].path,w_label,h_label));
        }
        cv::merge(array_of_labels,this_label);

        if (training_opts.image_augmentation) {
            auto [aug_img, aug_label]  = image_augmentation(this_img,this_label);

            for (int i = 0; i < aug_img.size(); i++) {
                img_tensor.push_back(convert_to_tensor(aug_img[i]));
                label_tensor.push_back(convert_to_tensor(aug_label[i]));
            }
        } else {
            img_tensor.push_back(convert_to_tensor(this_img));
            label_tensor.push_back(convert_to_tensor(this_label));
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

        auto experiment_path = data_path / entry;

        std::cout << "Loading experiment path: " << experiment_path << std::endl;

        // We may have different views (cameras) for the same experiment.
        for (const auto& this_view : data["images"]["views"]) {

            auto img_folder_path = generate_output_path_from_json(experiment_path, this_view["prefix"]);

            //The paths to all of our images will be found.
            std::vector<name_and_path> this_view_images = add_image_to_load(img_folder_path,data["images"]["filetypes"],data["images"]["name_prefix"]);

            //Our labels are located in this directory (or subdirects of this directory)
            auto label_folder_path = generate_output_path_from_json(experiment_path, this_view["label_prefix"]);

            // Our labels can be masks corresponding to each image, or points (which we will generate masks from)
            // image type should be directed to folders
            std::filesystem::path this_label_folder_path = label_folder_path;

            for (int i = 0; i< data["labels"]["labels"].size(); i ++ ) {

                std::string label_name = data["labels"]["labels"][i]["name"];
                auto label_filetypes = data["labels"]["labels"][i]["filetypes"];
                auto label_prefix = data["labels"]["labels"][i]["name_prefix"];
                std::string label_type = data["labels"]["labels"][i]["type"];

                this_label_folder_path /= label_name;
                std::vector<name_and_path> this_view_labels;

                if (label_type.compare("mask") == 0) {
                    this_view_labels = add_image_to_load(this_label_folder_path,label_filetypes,
                                    label_prefix);
                } else {
                    std::cout << "unsupported filetype for label. aborting" << std::endl;
                    break;
                }

                // Loop through all of the images
                for (const auto& this_img : this_view_images) {

                    // Loop through all of the labels 
                    for (const auto& this_label : this_view_labels) {
                        if (this_img.name.compare(this_label.name) == 0) {
                            img_label_files.push_back(img_label_pair(this_img.path,this_label.path));
                            break;
                        }
                    }
                }

            }
        }  
    }

    std::cout << "The total number of images is " << img_label_files.size() << std::endl;

    return img_label_files;
};

/////////////////////////////////////////////////////////////////////////////////

 class MyDataset : public torch::data::Dataset<MyDataset>
{
    private:
        torch::Tensor states_, labels_;

    public:
        explicit MyDataset(training_options& training_opts) 
        {
            auto img_label_files = read_json_file(training_opts.config_file);

            auto [states, labels] = read_images(img_label_files, training_opts);

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
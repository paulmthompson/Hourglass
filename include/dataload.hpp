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
#include <unordered_map>

#include "augmentation.hpp"

using namespace torch;
namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace cv;

using paths = std::vector<std::filesystem::path>;

#pragma once

//https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-3-advanced-libtorch/
cv::Mat load_image_from_path(const std::filesystem::path& image_path, int w, int h) {
    if (!fs::exists(image_path)) {
        std::cout << "File path does not exist at " << image_path.string() << std::endl;
    }
    cv::Mat raw_image = cv::imread(image_path.string(),cv::IMREAD_GRAYSCALE);
    cv::Mat image;
    cv::resize(raw_image, image,cv::Size(w,h), cv::INTER_AREA);

    if (!image.isContinuous()) {   
        image = image.clone(); 
    }

    return image;
};

cv::Mat generate_heatmap(int x, int y, const int rad, const int w, const int h) {
    
    int img_w = 640;
    int img_h = 480;

    if ((x > img_w) | (y > img_h)) {
        std::cout << "Coordinate is out of bounds" << std::endl;
    }

    cv::Mat raw_image = cv::Mat::zeros(img_h,img_w,CV_32FC1);
    float & point = raw_image.at<float>(y,x);
    point = 255.0;

    cv::Mat raw_image2;
    cv::GaussianBlur(raw_image, raw_image2, cv::Size(rad,rad), 0);

    cv::Mat raw_image3;
    cv::normalize(raw_image2,raw_image3,0.0,255.0,cv::NORM_MINMAX);

    cv::Mat image = cv::Mat(h,w,CV_32FC1);

    cv::resize(raw_image3,image,image.size(),0.0,0.0,cv::INTER_AREA);

    cv::Mat image_out = cv::Mat(h,w,CV_8UC1);
    image.convertTo(image_out,CV_8UC1);

    if (!image_out.isContinuous()) {   
        image_out = image_out.clone(); 
    }

    imwrite("test.png",image_out);

    return image_out;
}

/////////////////////////////////////////////////////////////////////////////////
class label_path {
public:
label_path() = default;

label_path(const label_path&) =delete;
void operator=(const label_path&) =delete;

virtual ~label_path() {}

virtual cv::Mat load_image(int w, int h) const = 0;

};

class pixel_label_path : public label_path {
public:
    pixel_label_path(int x, int y,int rad) {
        this->x = x;
        this->y = y;
        this->rad = rad; // This needs to be an odd number
    }
    cv::Mat load_image(int w, int h) const override {
        return generate_heatmap(this->x, this->y, this->rad, w, h);
    }

private:
int x;
int y;
int rad;

};

class img_label_path : public label_path {
    public:

    img_label_path(std::filesystem::path this_path) {
        this->path = this_path;
    }

    cv::Mat load_image(int w, int h) const override {
        return load_image_from_path(this->path,w,h);
    }

    private:
    std::filesystem::path path;

};

class img_label_pair {
    public:
    img_label_pair() = default;
    img_label_pair(fs::path this_img) {
        this->img = this_img;
        //this->labels = std::vector<std::unique_ptr<label_path>>();
    }
    img_label_pair(fs::path this_img, fs::path this_label) {
        this->img = this_img;
        this->labels = std::vector<std::unique_ptr<label_path>>();
        this->labels.emplace_back(std::make_unique<img_label_path>(this_label));
    }
    void add_label(fs::path this_label) {
        auto label = std::make_unique<img_label_path>(this_label);
        this->labels.push_back(std::unique_ptr<label_path>(std::move(label)));
    }
    void add_label(int x, int y,int rad) {
        auto label = std::make_unique<pixel_label_path>(x,y,rad);
        this->labels.push_back(std::unique_ptr<label_path>(std::move(label)));
    }
    fs::path img;
    std::vector<std::unique_ptr<label_path>> labels;
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
        this->keypoint_radius = 101;
        if (data["training"].contains("keypoint-radius")) {
            this->keypoint_radius = data["training"]["keypoint-radius"];
        }

        this->config_file = config_file;
    }

    int batch_size;
    int epochs;
    float learning_rate;
    bool image_augmentation;
    std::string config_file;
    int keypoint_radius;
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

/////////////////////////////////////////////////////////////////////////////////

typedef enum {MASK, PIXEL} LABEL_TYPE;

struct name_and_path {
    std::string name;
    std::filesystem::path path;
    int x;
    int y;
    LABEL_TYPE label_type;
    name_and_path() = default;
    name_and_path(std::string name, std::filesystem::path path) {
        this->name = name;
        this->path = path;
        this->x = 0;
        this->y = 0;
        this->label_type = MASK;
    }
    name_and_path(std::string name, int x, int y) {
        this->name = name;
        this->path = "";
        this->x = x;
        this->y = y;
        this->label_type = PIXEL;
    }
};

//I think this should be an unordered map

std::unordered_map<std::string,name_and_path> 
add_image_to_load(const std::filesystem::path& folder_path, const json& json_filetypes, const json& json_prefix) {
    
    std::unordered_map<std::string,name_and_path> out_images;

    std::regex image_regex(json_prefix);
    for (const auto & file_type : json_filetypes) {
        for (const auto & entry : fs::directory_iterator(folder_path)) {
            if (entry.path().extension() == file_type.get<std::string>()) {
                std::filesystem::path image_file_path = folder_path / entry.path();

                // Here we remove the image name prefix so that it is just a number.
                std::string image_name = std::regex_replace(entry.path().stem().string(), image_regex, "");
                out_images[image_name]=name_and_path(image_name,image_file_path);
            }
        }
    }
    return out_images;
};

//Unordered map with image name as key
std::unordered_map<std::string,name_and_path> 
add_pixels_to_load(const std::filesystem::path& folder_path,const std::string& img_prefix, std::string label_name) {
    
    std::unordered_map<std::string,name_and_path> out_images;

    std::filesystem::path label_path;
    for (const auto & entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".json") {
            label_path = entry.path();
            std::cout << entry.path().string() << std::endl;
        }
    }
    
    std::ifstream f(label_path);
    json data = json::parse(f);
    f.close();

    std::regex image_regex(img_prefix);
    for (const auto& label : data) {
        fs::path img_name = label["image"];
        std::string img_name2 = std::regex_replace(img_name.stem().string(), image_regex, "");
        int x = label["labels"][label_name][0]; // X
        int y = label["labels"][label_name][1]; // Y

        out_images[img_name]=name_and_path(img_name2,x,y);
    }
    
    
    return out_images;
};


/////////////////////////////////////////////////////////////////////////////////

torch::Tensor convert_to_tensor(cv::Mat& image) {

    auto tensor = torch::empty(
           { image.rows, image.cols, image.channels()},
            torch::TensorOptions()  
               .dtype(torch::kByte)   
               .device(torch::kCPU));     
               
    // Copy over the data 
    std::memcpy(tensor.data_ptr(), reinterpret_cast<void*>(image.data), tensor.numel() * sizeof(at::kByte));

    return tensor.permute({2,0,1}); // Single image
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
        /*
        for (int i = 1; i < 2; i++) {
            array_of_labels.push_back(this_img_label.labels[i]->load_image(w_label,h_label));
        }
        */
        
        for (auto const& f : this_img_label.labels) {
            array_of_labels.push_back(f->load_image(w_label,h_label));
        }
        
        cv::merge(array_of_labels,this_label);

        //std::cout << "Image width: " << this_label.rows << std::endl;
        //std::cout << "Image height: " << this_label.cols << std::endl;
        //std::cout << "Image channels: " << this_label.channels() << std::endl;

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

 /////////////////////////////////////////////////////////////////////////////////

 std::optional<std::unordered_map<std::string,name_and_path>> 
 get_labels_name_and_path(const fs::path& this_label_folder_path, 
                                                const std::string& label_name, 
                                                const std::string& label_type,
                                                const int label_num,
                                                const json& data) {

    std::unordered_map<std::string,name_and_path> this_view_labels; // This should be an unordered map 

    if (label_type.compare("mask") == 0) {

        auto label_filetypes = data["labels"]["labels"][label_num]["filetypes"];
        auto label_prefix = data["labels"]["labels"][label_num]["name_prefix"];

        this_view_labels = add_image_to_load(this_label_folder_path,label_filetypes,
                                    label_prefix);
    } else if (label_type.compare("pixel") == 0) {

        auto label_prefix = data["labels"]["labels"][label_num]["name_prefix"];
        this_view_labels = add_pixels_to_load(this_label_folder_path,label_prefix, label_name);
    } else {
        std::cout << "unsupported filetype for label. aborting" << std::endl;
        return std::nullopt;
    }

    return this_view_labels;
 };

void match_image_and_labels(std::vector<name_and_path>& this_view_images, 
                            std::vector<name_and_path>& this_view_labels) {

    std::vector<img_label_pair> img_label_files;

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
};

std::vector<img_label_pair> read_json_file(training_options& opts) {

    std::ifstream f(opts.config_file);
    json data = json::parse(f);
    f.close();

    std::cout << "Configuration file found with name " << data["name"] << std::endl;

    std::filesystem::path data_path = data["folder_path"];

    std::vector<img_label_pair> img_label_files; // Output for all views and experiments

    for (const auto& entry : data["experiments"]) {

        auto experiment_path = data_path / entry;

        std::cout << "Loading experiment path: " << experiment_path << std::endl;

        // We may have different views (cameras) for the same experiment.
        for (const auto& this_view : data["images"]["views"]) {

            auto img_folder_path = generate_output_path_from_json(experiment_path, this_view["prefix"]);

            //The paths to all of our images will be found.
            auto this_view_images = add_image_to_load(img_folder_path,data["images"]["filetypes"],data["images"]["name_prefix"]);

            //Our labels are located in this directory (or subdirects of this directory)
            auto label_folder_path = generate_output_path_from_json(experiment_path, this_view["label_prefix"]);

            // Our labels can be masks corresponding to each image, or points (which we will generate masks from)
            // image type should be directed to folders
            std::filesystem::path this_label_folder_path = label_folder_path;

            std::vector<std::unordered_map<std::string, name_and_path>> view_labels;

            for (int i = 0; i< data["labels"]["labels"].size(); i ++ ) {

                std::string label_name = data["labels"]["labels"][i]["name"];
                std::string label_type = data["labels"]["labels"][i]["type"];

                auto this_label_folder_path = label_folder_path / label_name;
                
                auto this_view_labels = get_labels_name_and_path(this_label_folder_path,
                                                label_name, 
                                                label_type,
                                                i,
                                                data);

                if (this_view_labels.has_value()) {
                    view_labels.push_back(this_view_labels.value());
                }
            }
            // Loop through all of the images
            for (const auto& this_img : this_view_images) {

                std::vector<name_and_path> matched_labels;
                // Loop through all of the labels types
                for (int i = 0; i < view_labels.size(); i ++) {

                    for (const auto& this_label : view_labels[i]) {
                        if (this_img.second.name.compare(this_label.second.name) == 0) {
                            matched_labels.push_back(this_label.second);
                            break;
                        }
                    }
                }
                

                if (matched_labels.size() == view_labels.size()) {
                    img_label_files.push_back(img_label_pair(this_img.second.path));
                    for (const auto& label : matched_labels) {
                        if (label.label_type == MASK) {
                            img_label_files.back().add_label(label.path);
                        } else if (label.label_type == PIXEL){
                            img_label_files.back().add_label(label.x,label.y,opts.keypoint_radius);
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
            auto img_label_files = read_json_file(training_opts);

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
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

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
#include "training_options.hpp"

using namespace torch;
namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace cv;

using paths = std::vector<std::filesystem::path>;

#pragma once

#if defined _WIN32 || defined __CYGWIN__
	#define DLLOPT __declspec(dllexport)
#else
	#define DLLOPT __attribute__((visibility("default")))
#endif

/////////////////////////////////////////////////////////////////////////////////


 class MyDataset : public torch::data::Dataset<MyDataset>
{
    private:
        torch::Tensor states_, labels_;

    public:
        explicit MyDataset(training_options& training_opts);
        torch::data::Example<> get(size_t index) override;
        torch::optional<size_t> size() const override;
};

/////////////////////////////////////////////////////////////////////////////////

// Labels can take the form of 1) images (masks) or 2) pixel coordinates (x,y) which can be used to generate a heatmap image

class Label {

public:
    Label() = default;

    Label(const Label&) =delete;
    void operator=(const Label&) =delete;

    virtual ~Label() {}

    virtual cv::Mat load_image(int w, int h) const = 0;

private:
    cv::Mat resize_image(const cv::Mat& img, int w, int h) const;
};

class PixelLabel : public Label {
public:
    PixelLabel(int x, int y,int rad);
    cv::Mat load_image(int w, int h) const override;

private:
    int x;
    int y;
    int rad;
    cv::Mat generate_heatmap(int x, int y, const int rad, const int w, const int h) const;
};

class MaskLabel : public Label {
    
public:
    MaskLabel(std::filesystem::path this_path);
    cv::Mat load_image(int w, int h) const override;

private:
    std::filesystem::path path;
};

// An image can have multiple labels, so this class stores a vector of labels
class img_label_pair {

public:
    img_label_pair() = default;
    img_label_pair(fs::path this_img);
    img_label_pair(fs::path this_img, fs::path this_label);
    void add_label(fs::path this_label);
    void add_label(int x, int y,int rad);

    fs::path img;
    std::vector<std::unique_ptr<Label>> labels;
};

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

/////////////////////////////////////////////////////////////////////////////////

//https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-3-advanced-libtorch/
inline cv::Mat load_image_from_path(const std::filesystem::path& image_path, int w, int h) {
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

/////////////////////////////////////////////////////////////////////////////////

//This helper method takes a vector of tensors and then stacks them into a single tensor.
//It also converts the tensor to float and normalizes it to 0-1. (Should this be checked for?)

inline torch::Tensor make_tensor_stack(std::vector<torch::Tensor>& tensor) {
    auto stacked = torch::stack(torch::TensorList(tensor));

    return stacked.to(torch::kFloat32).div(255);
}

template<typename T>
inline void shuffle(std::vector<T>& imgs, std::vector<T>& labels) {

    //TODO check that the vectors are the same size
    if (imgs.size() != labels.size()) {
        std::cout << "The label and image vectors are not the same size" << std::endl;
        std::cout << "Images size: " << imgs.size() << std::endl;
        std::cout << "Labels size: " << labels.size() << std::endl;
        std::cout << "Aborting shuffle" << std::endl;
    }

    int n = imgs.size();
    //https://www.techiedelight.com/shuffle-vector-cpp/
    for (int i = 0; i < n - 1; i++)
    {
        int j = i + rand() % (n - i);
        std::swap(imgs[i],imgs[j]);
        std::swap(labels[i],labels[j]);
    }
};

/////////////////////////////////////////////////////////////////////////////////

inline torch::Tensor convert_to_tensor(cv::Mat& image) {

    auto tensor = torch::empty(
           { image.rows, image.cols, image.channels()},
            torch::TensorOptions()  
               .dtype(torch::kByte)   
               .device(torch::kCPU));     
               
    // Copy over the data 
    std::memcpy(tensor.data_ptr(), reinterpret_cast<void*>(image.data), tensor.numel() * sizeof(at::kByte));

    return tensor.permute({2,0,1}); // Single image
}

inline std::tuple<int,int> get_width_height(const std::string& config_file, const std::string& keyword) {

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    std::vector<int> height_width = data[keyword]["resolution"];

    return std::make_tuple(height_width[0],height_width[1]);
};

 //https://discuss.pytorch.org/t/libtorch-how-to-use-torch-datasets-for-custom-dataset/34221/2
 inline std::tuple<torch::Tensor,torch::Tensor> read_images(const std::vector<img_label_pair>& image_paths, training_options& training_opts)
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

inline void match_image_and_labels(std::vector<name_and_path>& this_view_images, 
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

/////////////////////////////////////////////////////////////////////////////////

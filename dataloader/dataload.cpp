#include "dataload.hpp"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>

using json = nlohmann::json;
using namespace torch;


training_options::training_options(const std::string& config_file) {
    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    if (data["training"].contains("epochs")) {
         this->epochs = data["training"]["epochs"];
    }

    if (data["training"].contains("batch-size")) {
        this->batch_size = data["training"]["batch-size"];
    }

    if (data["training"].contains("learning-rate")) {
        this->learning_rate = data["training"]["learning-rate"];
    }

    if (data["training"].contains("image-augmentation")) {
         this->image_augmentation = data["training"]["image-augmentation"];
    }

    if (data["training"].contains("keypoint-radius")) {
        this->keypoint_radius = data["training"]["keypoint-radius"];
    }

    if (data["training"].contains("save-name")) {
        this->weight_save_name = data["training"]["save-name"];
    }

    if (data["training"]["load-weights"].contains("load")) {
        this->load_weights = data["training"]["load-weights"]["load"];
    }

    if (data["training"]["load-weights"].contains("path")) {
        this->load_weight_path = data["training"]["load-weights"]["path"];
    }

    if (data["training"].contains("intermediate-supervision")) {
        this->intermediate_supervision = data["training"]["intermediate-supervision"];
    }

    this->config_file = config_file;
}

/////////////////////////////////////////////////////////////////////////////////

cv::Mat Label::resize_image(const cv::Mat& raw_image, int w, int h) const {

    cv::Mat image;
    cv::resize(raw_image, image,cv::Size(w,h), cv::INTER_AREA);

    if (!image.isContinuous()) {   
        image = image.clone(); 
    }

    return image;
};

PixelLabel::PixelLabel(int x, int y,int rad) {
    this->x = x;
    this->y = y;
    this->rad = rad; // This needs to be an odd number
}

cv::Mat PixelLabel::load_image(int w, int h) const {
    return this->generate_heatmap(this->x, this->y, this->rad, w, h);
}

 // This takes a 2D coordinate and generates a heatmap with a gaussian distribution
// The keypoint coordinate (x,y) is from an an image of size (img_w,img_h)
// The heatmap is generated with size (w,h)

cv::Mat PixelLabel::generate_heatmap(int x, int y, const int rad, const int w, const int h) const {
    
    int img_w = 640;
    int img_h = 480;

    if ((x > img_w) | (y > img_h)) {
        std::cout << "Coordinate is out of bounds" << std::endl;
    }

    cv::Mat raw_image = cv::Mat::zeros(img_h,img_w,CV_32FC1);
    if ((x >= 0) & (y >= 0)) {
        float & point = raw_image.at<float>(y,x);
        point = 255.0;
    } else {
        std::cout << "Label at (0,0) is interpreted as no label for this image" << std::endl;
    }

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

    //imwrite("test.png",image_out);

    return image_out;
};

MaskLabel::MaskLabel(std::filesystem::path this_path) {
    this->path = this_path;
}

cv::Mat MaskLabel::load_image(int w, int h) const {
    return load_image_from_path(this->path,w,h);
}

img_label_pair::img_label_pair(fs::path this_img) {
    this->img = this_img;
    //this->labels = std::vector<std::unique_ptr<label_path>>();
}

img_label_pair::img_label_pair(fs::path this_img, fs::path this_label) {
    this->img = this_img;
    this->labels = std::vector<std::unique_ptr<Label>>();
    this->labels.emplace_back(std::make_unique<MaskLabel>(this_label));
}

void img_label_pair::add_label(fs::path this_label) {
    auto label = std::make_unique<MaskLabel>(this_label);
    this->labels.push_back(std::unique_ptr<Label>(std::move(label)));
}

void img_label_pair::add_label(int x, int y,int rad) {
    auto label = std::make_unique<PixelLabel>(x,y,rad);
    this->labels.push_back(std::unique_ptr<Label>(std::move(label)));
}

/////////////////////////////////////////////////////////////////////////////////

//https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
std::optional<std::string> match_folder_in_path(const fs::path& dir_path,std::string folder_path) {

    for (const auto & entry : fs::directory_iterator(dir_path)) {
        if (entry.path().string().find(folder_path) != string::npos) {
            return entry.path().string();
        }
    }
    return std::nullopt;
};

/////////////////////////////////////////////////////////////////////////////////

std::optional<std::filesystem::path> generate_output_path_from_json(const std::filesystem::path& folder_path, const json& subpaths_json) {

    std::filesystem::path output_path = folder_path;

    for (const auto& sub_path : subpaths_json) {
        std::optional<std::string> matched_folder = match_folder_in_path(output_path,sub_path);
        if (matched_folder) {
            output_path /= matched_folder.value();
        } else {
            std::cout << "Did not find sub_path " << sub_path << std::endl;
            return std::nullopt;
        }
    }
    return output_path;
};


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
                out_images[image_name] = name_and_path(image_name,image_file_path);
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
        int x, y;
        if (label["labels"][label_name].size() > 0) {
            x = label["labels"][label_name][0]; // X
            y = label["labels"][label_name][1]; // Y
        } else {
            x = -1;
            y = -1;
        }
        out_images[img_name.string()] = name_and_path(img_name2,x,y);
    }
    
    
    return out_images;
};

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

            if (!img_folder_path.has_value()) {
                continue;
            }

            //The paths to all of our images will be found.
            auto this_view_images = add_image_to_load(img_folder_path.value(),data["images"]["filetypes"],data["images"]["name_prefix"]);

            //Our labels are located in this directory (or subdirects of this directory)
            auto label_folder_path = generate_output_path_from_json(experiment_path, this_view["label_prefix"]);

            if (!label_folder_path.has_value()) {
                continue;
            }

            // Our labels can be masks corresponding to each image, or points (which we will generate masks from)
            // image type should be directed to folders
            std::filesystem::path this_label_folder_path = label_folder_path.value();

            std::vector<std::unordered_map<std::string, name_and_path>> view_labels;

            for (int i = 0; i< data["labels"]["labels"].size(); i ++ ) {

                std::string label_name = data["labels"]["labels"][i]["name"];
                std::string label_type = data["labels"]["labels"][i]["type"];

                auto this_label_folder_path = label_folder_path.value() / label_name;
                
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

MyDataset::MyDataset(training_options& training_opts) 
{
    auto img_label_files = read_json_file(training_opts);

    auto [states, labels] = read_images(img_label_files, training_opts);

    states_ = std::move(states);
    labels_ = std::move(labels);

    std::cout << "Image size is " << states_.size(0) << std::endl;
    std::cout << "Label size is " << labels_.size(0) << std::endl;
}

torch::data::Example<> MyDataset::get(size_t index)
{
    // You may for example also read in a .csv file that stores locations
    // to your data and then read in the data at this step. Be creative.
    return {states_[index], labels_[index]};
}

torch::optional<size_t> MyDataset::size() const { return states_.size(0); }
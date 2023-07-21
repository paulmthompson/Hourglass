#include "training_options.hpp"

#include <nlohmann/json.hpp>

#include <fstream>

using json = nlohmann::json;

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

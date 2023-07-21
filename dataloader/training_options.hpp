#ifndef TRAINING_OPTIONS_HPP
#define TRAINING_OPTIONS_HPP

#include <string>

class training_options {

public:
    training_options(const std::string& config_file);

    training_options() : 
        batch_size(32), 
        epochs(1), 
        learning_rate(5e-5), 
        image_augmentation(false), 
        config_file(""), 
        keypoint_radius(101), 
        weight_save_name("hourglass_weights.pt"), 
        intermediate_supervision(true), 
        load_weights(false), 
        load_weight_path("hourglass_weights.pt") 
        {}

    int batch_size;
    int epochs;
    float learning_rate;
    bool image_augmentation;
    std::string config_file;
    int keypoint_radius;
    std::string weight_save_name;
    bool intermediate_supervision;
    bool load_weights;
    std::string load_weight_path;
 
};


#endif  // TRAINING_OPTIONS_HPP
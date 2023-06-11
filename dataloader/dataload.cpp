#include "dataload.hpp"

/////////////////////////////////////////////////////////////////////////////////

pixel_label_path::pixel_label_path(int x, int y,int rad) {
    this->x = x;
    this->y = y;
    this->rad = rad; // This needs to be an odd number
}

cv::Mat pixel_label_path::load_image(int w, int h) const {
    return generate_heatmap(this->x, this->y, this->rad, w, h);
}

img_label_path::img_label_path(std::filesystem::path this_path) {
    this->path = this_path;
}

cv::Mat img_label_path::load_image(int w, int h) const {
    return load_image_from_path(this->path,w,h);
}

img_label_pair::img_label_pair(fs::path this_img) {
    this->img = this_img;
    //this->labels = std::vector<std::unique_ptr<label_path>>();
}

img_label_pair::img_label_pair(fs::path this_img, fs::path this_label) {
    this->img = this_img;
    this->labels = std::vector<std::unique_ptr<label_path>>();
    this->labels.emplace_back(std::make_unique<img_label_path>(this_label));
}

void img_label_pair::add_label(fs::path this_label) {
    auto label = std::make_unique<img_label_path>(this_label);
    this->labels.push_back(std::unique_ptr<label_path>(std::move(label)));
}

void img_label_pair::add_label(int x, int y,int rad) {
    auto label = std::make_unique<pixel_label_path>(x,y,rad);
    this->labels.push_back(std::unique_ptr<label_path>(std::move(label)));
}

/////////////////////////////////////////////////////////////////////////////////
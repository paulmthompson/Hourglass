#include "utilities.hpp"

//https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-3-advanced-libtorch/
cv::Mat load_image_from_path(const std::filesystem::path& image_path, int w, int h) {
    if (!std::filesystem::exists(image_path)) {
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

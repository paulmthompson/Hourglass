#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <ffmpeg_wrapper/videodecoder.h>
#include <ffmpeg_wrapper/videoencoder.h>

#include "hourglass.hpp"
#include "dataloader/dataload.hpp"
#include "saveload.hpp"

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>  // for high_resolution_clock
#include <memory>
#include <array>
#include <filesystem>
#include <cstdlib>

using namespace torch;
using json = nlohmann::json;

#pragma once

/////////////////////////////////////////////////////////////////////////////////

class prediction_options {
public:
    prediction_options(const std::string& config_file) {
        std::ifstream f(config_file);
        json data = json::parse(f);
        f.close();

        this->vid_name = data["prediction"]["videos"]; // This should be made into an array of video properties that contain start frame and end frame
        std::cout << this->vid_name << std::endl;

        this->save_images = false;
        if (data["prediction"].contains("save_images")) {
            this->save_images = data["prediction"]["save_images"];
        }

        this->save_hdf5 = true;
        if (data["prediction"].contains("save_hdf5")) {
            this->save_hdf5 = data["prediction"]["save_hdf5"];
        }
        
        this->save_video = true;
        if (data["prediction"].contains("save_video")) {
            this->save_video = data["prediction"]["save_video"];
        }
        
        std::filesystem::path vid_path = this->vid_name;
        this->output_save_path = vid_path.stem().string();
        if (data["prediction"].contains("output_file_path")) {
            this->output_save_path = data["prediction"]["output_file_path"];
        }

        this->output_video_name = "./" + vid_path.stem().string() + "_labeled.mp4";

        this->starting_frame = 1;
        if (data["prediction"].contains("start_frame")) {
            if (data["prediction"]["start_frame"] > 0) {
                this->starting_frame = data["prediction"]["start_frame"];
            } else {
                this->starting_frame = 1;
            }
        }

        this->batch_size = 32;
        if (data["prediction"].contains("batch-size")) {
            this->batch_size = data["prediction"]["batch-size"];
        }

        this->total_images = 0;
        if (data["prediction"].contains("end_frame")) {
            this->total_images = data["prediction"]["end_frame"];
        }

        const int output_dims = data["hourglass"]["output-dimensions"];

        this->label_types = std::vector<LABEL_TYPE>(output_dims,LABEL_TYPE::MASK);
        if (data["hourglass"].contains("output-type")) {
            for (int i=0; i< output_dims; i++) {
                std::string label_type = data["hourglass"]["output-type"][i];
                if (label_type.compare("mask") == 0) {
                    this->label_types[i] = LABEL_TYPE::MASK;
                } else if (label_type.compare("pixel") == 0) {
                    this->label_types[i] = LABEL_TYPE::PIXEL;
                } else {
                    std::cout << "Do not recognize output type specified" << std::endl;
                }
            }
        }
        this->label_colors = std::vector<std::array<uint8_t,3>>(output_dims,std::array<uint8_t,3>{0,0,255});
        if (output_dims > 1) {
            this->label_colors[1] = {0,255,0};
        }
        if (data["prediction"].contains("label-colors")) {
            int label_index = 0;
            for (auto label_color : data["prediction"]["label-colors"]) {
                std::array<uint8_t,3> color = {label_color[0],label_color[1],label_color[2]};
                this->label_colors[label_index] = color;
                label_index += 1;
            }
        }

        this->load_weights = false;
        if (data["prediction"]["load-weights"].contains("load")) {
            this->load_weights = data["prediction"]["load-weights"]["load"];
        }
        this->load_weight_path = "hourglass_weights.pt";
        if (data["prediction"]["load-weights"].contains("path")) {
            this->load_weight_path = data["prediction"]["load-weights"]["path"];
        }

    }

    void update_total_images(const int num) {
        if (this->total_images == 0) {
            std::cout << "No ending frame specified, so analyzing to the end of the video" << std::endl;
            this->total_images = num;
        } else if (this->total_images >= num) {
            this->total_images = num;
        }
        if (this->total_images <= this->starting_frame) {
            std::cout << "Error: the start of the block of frames to analyze is after the end of that block. Check config file" << std::endl;
        }
    }
    int get_total_images() const {
        return this->total_images - this->starting_frame + 1;
    }
    std::string vid_name;
    bool save_images;
    bool save_hdf5;
    bool save_video;
    std::string output_save_path;
    std::string output_video_name;
    int64_t starting_frame;
    int batch_size;
    int64_t total_images;
    std::vector<LABEL_TYPE> label_types;
    std::vector<std::array<uint8_t,3>> label_colors;
    bool load_weights;
    std::string load_weight_path;
private:
    
};

/////////////////////////////////////////////////////////////////////////////////

inline torch::Tensor LoadFrame(ffmpeg_wrapper::VideoDecoder& vd, int frame_id) {

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

inline torch::Tensor LoadFrames(ffmpeg_wrapper::VideoDecoder& vd, int frame_start, int frame_end) {

    std::vector<torch::Tensor> frames;

    for (int i = frame_start; i <= frame_end; i++) {
        frames.push_back(LoadFrame(vd,i));
    }

    //std::cout << "Loaded frames " << frame_start << " - " << frame_end << std::endl;

    return make_tensor_stack(frames);
}

/////////////////////////////////////////////////////////////////////////////////

inline torch::Tensor prepare_for_opencv(torch::Tensor tensor,const int height, const int width) {

    tensor = nn::functional::interpolate(tensor,
        nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({height,width})).mode(torch::kBilinear).align_corners(false));

    tensor = tensor.mul(255).clamp(0,255).to(torch::kU8);

    tensor = tensor.detach().permute({2,3,1,0});
        
    return tensor.to(kCPU);
}

inline torch::Tensor get_hourglass_predictions(StackedHourglass &hourglass, torch::Tensor& data,const int height, const int width) {
    std::vector<torch::Tensor> output = hourglass->forward(data);

    torch::Tensor prediction = output.back();
        
    return prepare_for_opencv(prediction,height, width);
}
//This isn't quite right because I need to scale the other pixels 
inline cv::Mat combine_overlay(const cv::Mat& img, const cv::Mat& label,
                        const std::array<bool,3> color = {false, false, true}) {
    
    cv::Mat color_img;
    cv::Mat color_label;

    cv::Mat channel[3];
    cv::Mat dst;

    cv::cvtColor(img,color_img,cv::COLOR_GRAY2RGB);
    cv::cvtColor(label,color_label,cv::COLOR_GRAY2RGB);

    cv::split(color_label,channel);

    for (int i = 0; i < 3; i++) {
        if (!color[i]) {
            channel[i] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1); 
        }
    }

    cv::merge(channel,3,color_label);

    cv::addWeighted(color_label,0.5, color_img,0.5,0.0,dst);

    return dst;
}

/*
Here we are taking a black and white input (img), 
and creating a color image (color_img) by overlaying the labels (labels) on top of it.

To make the labels a color, we want to create a duplicate of the grayscale image, and then
set the color channels to 0 or 255 depending on the color array.

If we use a "black" background color, we end up with a significantly darkened image
https://stackoverflow.com/questions/67989210/blending-images-without-color-change
https://docs.opencv.org/3.4/d0/d86/tutorial_py_image_arithmetics.html
*/
inline cv::Mat combine_overlay(const cv::Mat& img, const std::vector<cv::Mat>& labels,
                        const std::vector<std::array<uint8_t,3>>& label_colors) {

    //This function should add labels on top of the input image and return the result   
    cv::Mat color_img;
    cv::cvtColor(img,color_img,cv::COLOR_GRAY2RGB); // Convert our input img to a color image

    // We will create a copy of the color image as our background for the labels
    cv::Mat label_img = color_img.clone();

    //We split the label image into its color channels.
    cv::Mat channel[3];
    cv::split(label_img,channel);

    for (int j=0; j< labels.size(); j++) {

        //For each label we are going to generate a mask of the label
        //By setting those pixels to an opaque color.
        const double label_threshold = 128; // Labels above this value will be set to label value
        const double label_maxval = 255; // Labels above threshold will be set to this value. 
        cv::Mat label_mask; // Label mask will be all black except for label pixels which are white
        cv::threshold(labels[j], label_mask, label_threshold, label_maxval,cv::THRESH_BINARY);

        //For the color assigned to the label, we set the color channel(s) to the value of the label
        for (int i = 0; i < 3; i++) {
            //if (label_colors[j][i]) {
                channel[i].setTo(label_colors[j][i],label_mask); // This will make all pixels in the label mask opaque in that channel
                //cv::bitwise_or(channel[i],labels[j],channel[i],label_mask); // This will make all pixels in the label mask opaque in that channel
            //}
        }
    }

    //Now merge the channels back together to get an image with opaque colors drawn on top.
    cv::Mat color_label;
    cv::merge(channel,3,color_label);

    //We can blend the opaque image with our unlabeled image to add some transparency.
    cv::addWeighted(color_label,0.5, color_img,0.5,0.0,color_img);

    return color_img;
    
}

inline void get_data_to_save_mask(const torch::Tensor& pred, save_structure& save,const int frame_index,const int channel_index) {

    float thres = 0.1 * 255;

    for (int j = 0; j < pred.size(3); j++) {

        auto my_slice = pred.index({torch::indexing::Slice(),torch::indexing::Slice(),channel_index,j});
        
        if (torch::any(my_slice.greater(thres)).item().toBool()) {

            save.save_frame(my_slice,frame_index + j,thres);

        }
    }
};


inline void get_data_to_save_pixel(const torch::Tensor& pred, save_structure& save,const int frame_index,const int channel_index) {

    for (int j = 0; j < pred.size(3); j++) {

        auto my_slice = pred.index({torch::indexing::Slice(),torch::indexing::Slice(),channel_index,j});
        auto ind = my_slice.argmax().item().toLong();
        auto prob = my_slice.max().item().toFloat();

        save.save_keypoint(my_slice,frame_index + j,ind,prob);
    }
};


/////////////////////////////////////////////////////////////////////////////////

template <class T>
inline void predict(StackedHourglass &hourglass, T &data_set, torch::Device device, const std::string &config_file)
{

    torch::NoGradGuard no_grad; // Turn off autograd for inference
    hourglass->eval();
    hourglass->to(device);

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    std::filesystem::create_directory("images");

    int batch_size = data["prediction"]["batch-size"];
    const int64_t batches_per_epoch = std::ceil(data_set.size().value() / static_cast<double>(batch_size));
    const int64_t total_images = data_set.size().value();

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(data_set),
        batch_size);

    const int out_height = 256;
    const int out_width = 256;

    auto start = std::chrono::high_resolution_clock::now();

    const auto output_channels = hourglass->get_output_dims();

    float loss = 0.0;
    int64_t batch_index = 0;
    int64_t img_num = 0;
    for (auto &batch : *data_loader)
    {
        auto data = batch.data.to(device);
        auto labels = batch.target.to(device);

        auto prediction = get_hourglass_predictions(hourglass,data,out_height,out_width);

        auto tensor_raw_data_ptr = prediction.template data_ptr<uchar>();

        data = prepare_for_opencv(data,out_height,out_width);

        auto data_raw_data_ptr = data. template data_ptr<uchar>();

        int label_to_read = 0;
        for (int j = 0; j < prediction.size(3); j++) {

            cv::Mat realImg(out_height, out_width, CV_8UC1, data_raw_data_ptr + (out_height*out_width*j));

            for (int k = 0; k < output_channels; k++) {

                cv::Mat resultImg(out_height,out_width,CV_8UC1, tensor_raw_data_ptr + (out_height*out_width*label_to_read));

                resultImg = combine_overlay(realImg,resultImg);

                std::string img_name = "images/test" + std::to_string(img_num) + "_" + std::to_string(k) + ".png";
                cv::imwrite(img_name,resultImg);

                label_to_read += 1;
            }
            img_num += 1;
        }
      
        std::cout << "\r"
                    "["
                    << (++batch_index) * batch_size << "/" << total_images << "]"
                    << " Predicted" << std::flush;

    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - start;

    std::cout << std::endl;
    std::cout << total_images << " images predicted in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average " << total_images / elapsed.count() << " images per second" << std::endl;
};

inline void predict_video(StackedHourglass &hourglass, torch::Device device, const std::string &config_file)
{
    auto options = prediction_options(config_file);
    
    if (options.load_weights) {
        load_weights(hourglass,options.load_weight_path);
    }

    torch::NoGradGuard no_grad; // Turn off autograd for inference
    hourglass->eval();
    hourglass->to(device);
    
    std::filesystem::create_directory("images");

    auto vd = ffmpeg_wrapper::VideoDecoder();
    auto ve = ffmpeg_wrapper::VideoEncoder();
     
    if (!std::filesystem::exists(options.vid_name)) {
        std::cout << "Can not find video file. Is the file path correct?" << std::endl;
        std::cout << "Aborting Prediction" << std::endl;
        std::exit(1);
    }

    vd.createMedia(options.vid_name);
    options.update_total_images(vd.getFrameCount());

    const int64_t batches_per_epoch = std::ceil(options.total_images / static_cast<double>(options.batch_size));
    
    const int out_height = vd.getHeight();
    const int out_width = vd.getWidth();

    auto save_frame = std::vector<uint32_t>(out_height * out_width,0);
    if (options.save_video) {
        ve.setSavePath(options.output_video_name); // This makes it fail? Not making a good enough save name somehow
        ve.createContext(out_width,out_height,25);
        ve.set_pixel_format(ffmpeg_wrapper::VideoEncoder::INPUT_PIXEL_FORMAT::RGB0);
        ve.openFile();
    }

    auto start = std::chrono::high_resolution_clock::now();

    const auto output_channels = hourglass->get_output_dims();

    std::vector<save_structure> save_output;
    for (int i = 0; i < output_channels; i ++) {
        save_output.push_back(save_structure(out_height,out_width));
        if (options.label_types[i] == LABEL_TYPE::PIXEL) {
            save_output.back().init_keypoint();
        }
    }
    
    int64_t batch_index = 0;
    //Frames are 0 indexed by ffmpeg, but we specify frames as 1,2,3 etc

    int64_t frame_index = options.starting_frame;
    
    while (frame_index < options.total_images)
    {
        int last_index = frame_index + options.batch_size - 1;
        last_index = (last_index <= options.total_images) ? last_index : options.total_images;
        auto data = LoadFrames(vd,frame_index-1,last_index-1);

        data = data.to(device);

        auto data_input = nn::functional::interpolate(data,
            nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({256,256}))
            .mode(torch::kBilinear)
            .antialias(true)
            .align_corners(false));

        auto prediction = get_hourglass_predictions(hourglass,data_input,out_height,out_width);

        auto prediction_raw_data_ptr = prediction.data_ptr<uchar>();

        if (options.save_hdf5) {
            for (int i = 0; i < output_channels; i ++ ) {
                if (options.label_types[i] == LABEL_TYPE::MASK) {
                    get_data_to_save_mask(prediction,save_output[i],frame_index,i);
                } else if (options.label_types[i] == LABEL_TYPE::PIXEL) {
                    get_data_to_save_pixel(prediction,save_output[i],frame_index,i);
                }
            }
        }

        if (options.save_images | options.save_video) {

            data = prepare_for_opencv(data,out_height,out_width);

            auto data_raw_data_ptr = data.data_ptr<uchar>();

            int label_to_read = 0;
            for (int j = 0; j < prediction.size(3); j++) {

                cv::Mat realImg(out_height, out_width, CV_8UC1, data_raw_data_ptr + (out_height*out_width*j));
                std::vector<cv::Mat> resultImg;

                for (int k = 0; k < output_channels; k++) {

                    resultImg.push_back(cv::Mat(out_height,out_width,CV_8UC1, prediction_raw_data_ptr + (out_height*out_width*label_to_read)));
                    
                    label_to_read += 1;
                }

                cv::Mat overlayImg = combine_overlay(realImg,resultImg,options.label_colors);
            
                if (options.save_images) {

                    std::string img_name = "images/test" + std::to_string(frame_index + j) + ".png";
                    cv::imwrite(img_name,overlayImg);
                    
                } 
                if (options.save_video) {

                    cv::Mat overlayImg2;
                    cv::cvtColor(overlayImg,overlayImg2,cv::COLOR_RGB2BGRA); // Puts intensity in the red channel
                    std::memcpy(&save_frame.data()[0], reinterpret_cast<void*>(overlayImg2.data), out_height*out_width * sizeof(uint32_t));
                    ve.writeFrameRGB0(save_frame);
                }
            }
        }
          
        std::cout << "\r"
                    "["
                    << (++batch_index) * options.batch_size << "/" << options.get_total_images() << "]"
                    << " Predicted" << std::flush;
        frame_index = frame_index + options.batch_size;
    }

    if (options.save_video) {
        ve.closeFile();
    }

    std::cout << std::endl;
    for (int i = 0; i < output_channels; i ++) {
        std::string fn = options.output_save_path + "_" + std::to_string(i) + ".h5";
        save_output[i].write(fn);
    } // This should be done more frequently to ensure that RAM doesn't disappear.

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - start;

    std::cout << std::endl;
    std::cout << options.get_total_images() << " images predicted in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average " << (options.get_total_images()) / elapsed.count() << " images per second" << std::endl;

}
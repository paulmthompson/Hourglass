#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <ffmpeg_wrapper/videodecoder.h>
#include <ffmpeg_wrapper/videoencoder.h>

#include "hourglass.hpp"
#include "dataload.hpp"
#include "saveload.hpp"

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>  // for high_resolution_clock
#include <memory>

using namespace torch;
using json = nlohmann::json;

#pragma once

torch::Tensor prepare_for_opencv(torch::Tensor tensor,const int height, const int width) {

    tensor = nn::functional::interpolate(tensor,
        nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({height,width})).mode(torch::kNearest));

    tensor = tensor.mul(255).clamp(0,255).to(torch::kU8);

    tensor = tensor.detach().permute({2,3,1,0});
        
    return tensor.to(kCPU);
}

torch::Tensor get_hourglass_predictions(StackedHourglass &hourglass, torch::Tensor& data,const int height, const int width) {
    auto output = hourglass->forward(data);


    torch::Tensor prediction = output.back();
        
    return prepare_for_opencv(prediction,height, width);
}
//This isn't quite right because I need to scale the other pixels 
cv::Mat combine_overlay(const cv::Mat& img, const cv::Mat& label) {
    
    cv::Mat color_img;
    cv::Mat color_label;

    cv::Mat channel[3];
    cv::Mat dst;

    cv::cvtColor(img,color_img,cv::COLOR_GRAY2RGB);
    cv::cvtColor(label,color_label,cv::COLOR_GRAY2RGB);

    cv::split(color_label,channel);

    channel[0] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1); 
    channel[1] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

    cv::merge(channel,3,color_label);

    cv::addWeighted(color_label,0.5, color_img,0.5,0.0,dst);

    return dst;
}

class prediction_options {
public:
    prediction_options(const std::string& config_file) {
        std::ifstream f(config_file);
        json data = json::parse(f);
        f.close();

        this->vid_name = data["prediction"]["videos"];
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
        this->output_save_path = vid_path.stem().string() + ".h5";
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
private:
    
};

template <class T>
void predict(StackedHourglass &hourglass, T &data_set, torch::Device device, const std::string &config_file)
{
    torch::NoGradGuard no_grad; // Turn off autograd for inference

    std::ifstream f(config_file);
    json data = json::parse(f);
    f.close();

    hourglass->to(device);

    int batch_size = data["prediction"]["batch-size"];
    const int64_t batches_per_epoch = std::ceil(data_set.size().value() / static_cast<double>(batch_size));
    const int64_t total_images = data_set.size().value();

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(data_set),
        batch_size);

    load_weights(hourglass,config_file);

    const int out_height = 256;
    const int out_width = 256;

    auto start = std::chrono::high_resolution_clock::now();

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

        for (int j = 0; j < prediction.size(3); j++) {

            cv::Mat resultImg(out_height,out_width,CV_8UC1, tensor_raw_data_ptr + (out_height*out_width*j));
            cv::Mat realImg(out_height, out_width, CV_8UC1, data_raw_data_ptr + (out_height*out_width*j));

            resultImg = combine_overlay(realImg,resultImg);

            std::string img_name = "test" + std::to_string(img_num) + ".png";
            cv::imwrite(img_name,resultImg);
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

void get_data_to_save(const torch::Tensor& pred, save_structure& save,const int frame_index) {

    float thres = 0.1 * 255;

    for (int j = 0; j < pred.size(3); j++) {

        auto my_slice = pred.index({torch::indexing::Slice(),torch::indexing::Slice(),torch::indexing::Slice(),j});

        if (torch::any(my_slice.greater(thres)).item().toBool()) {
            //std::cout << "Tongue detected at " << frame_index + j << std::endl;
            save.save_frame(my_slice,frame_index + j,thres);
        }
    }
};

void predict_video(StackedHourglass &hourglass, torch::Device device, const std::string &config_file)
{
    torch::NoGradGuard no_grad; // Turn off autograd for inference

    auto options = prediction_options(config_file);
    
    hourglass->to(device);

    auto vd = ffmpeg_wrapper::VideoDecoder();
    auto ve = ffmpeg_wrapper::VideoEncoder();
     
    vd.createMedia(options.vid_name);
    options.update_total_images(vd.getFrameCount());

    const int64_t batches_per_epoch = std::ceil(options.total_images / static_cast<double>(options.batch_size));
    
    load_weights(hourglass,config_file);

    const int out_height = vd.getHeight();
    const int out_width = vd.getWidth();

    auto save_frame = std::vector<uint32_t>(out_height * out_width,0);
    if (options.save_video) {
        ve.setSavePath(options.output_video_name); // This makes it fail? Not making a good enough save name somehow
        ve.createContext(out_width,out_height,25);
        ve.set_pixel_format(ffmpeg_wrapper::VideoEncoder::INPUT_PIXEL_FORMAT::RGB0);
        ve.openFile();
    }

    auto save_output = save_structure(out_height,out_width);

    auto start = std::chrono::high_resolution_clock::now();

    int64_t batch_index = 0;
    //Frames are 0 indexed by ffmpeg, but we specify frames as 1,2,3 etc

    int64_t frame_index = options.starting_frame;
    
    while (frame_index < options.total_images)
    {
        int last_index = frame_index + options.batch_size - 1;
        last_index = (last_index <= options.total_images) ? last_index : options.total_images;
        auto data = LoadFrames(vd,frame_index-1,last_index-1);

        data = data.to(device);

        data = nn::functional::interpolate(data,
            nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({256,256})).mode(torch::kNearest));

        auto prediction = get_hourglass_predictions(hourglass,data,out_height,out_width);

        auto prediction_raw_data_ptr = prediction.data_ptr<uchar>();

        if (options.save_hdf5) {
            get_data_to_save(prediction,save_output,frame_index);
        }

        if (options.save_images | options.save_video) {

            data = prepare_for_opencv(data,out_height,out_width);

            auto data_raw_data_ptr = data.data_ptr<uchar>();

            for (int j = 0; j < prediction.size(3); j++) {

                cv::Mat resultImg(out_height,out_width,CV_8UC1, prediction_raw_data_ptr + (out_height*out_width*j));
                cv::Mat realImg(out_height, out_width, CV_8UC1, data_raw_data_ptr + (out_height*out_width*j));

                cv::Mat overlayImg = combine_overlay(realImg,resultImg);
            
                if (options.save_images) {

                    std::string img_name = "test" + std::to_string(frame_index + j) + ".png";
                    cv::imwrite(img_name,overlayImg);
                    
                } else if (options.save_video) {

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
    save_output.write(options.output_save_path); // This should be done more frequently to ensure that RAM doesn't disappear.

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - start;

    std::cout << std::endl;
    std::cout << options.get_total_images() << " images predicted in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average " << (options.get_total_images()) / elapsed.count() << " images per second" << std::endl;

}
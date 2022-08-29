#include <torch/torch.h>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#include <H5Cpp.h>

#include <vector>
#include <iostream>
#include <chrono> // for high_resolution_clock
#include <filesystem>
#include <string>

using namespace torch;
using namespace HighFive;

// The saving and loading of vectors of vectors in HDF5 was adapting from very useful discussions in the Highfive github page here:
// https://github.com/BlueBrain/HighFive/issues/369

class save_structure
{
public:
    save_structure(int x_dim, int y_dim)
    {
        this->x_dim = x_dim;
        this->y_dim = y_dim;
        this->prediction_frame = std::vector<uint8_t>(x_dim * y_dim);
    }
    void save_frame(const torch::Tensor &pred_frame, const int frame_index, const float thres) // save_frame is 4-D array
    {

        this->frame.push_back(frame_index);
        this->x_pos.push_back({});
        this->y_pos.push_back({});
        this->prob.push_back({});

        const int x_dim = pred_frame.size(0);
        const int y_dim = pred_frame.size(1);

        memcpy(&this->prediction_frame.data()[0], pred_frame.contiguous().data_ptr<uchar>(), x_dim * y_dim);

        for (int i = 0; i < this->prediction_frame.size(); i++)
        {
            if (this->prediction_frame[i] > thres)
            {
                this->x_pos.back().push_back(i / y_dim);
                this->y_pos.back().push_back(i % y_dim);
                this->prob.back().push_back((float) this->prediction_frame[i] / 255.0);
            }
        }
    }
    void write(std::string &path)
    {

        H5Easy::File file_frame(path.c_str(), H5Easy::File::Overwrite);

        H5Easy::dump(file_frame, "/frames", this->frame);
        file_frame.flush();

        H5::H5File file(file_frame.getId());

        write_var_length_array(file,std::string("heights"),this->x_pos);
        write_var_length_array(file,std::string("widths"),this->y_pos);
        write_var_length_array(file,std::string("probs"),this->prob);

        std::cout << "Data saved as " << path << std::endl;
    }

private:

    template <typename T>
    std::vector<hvl_t> make_varlen(const hsize_t n_rows,std::vector<std::vector<T>>& input_vector) {
        std::vector<hvl_t> varlen_spec(n_rows);

        for (hsize_t idx = 0; idx < n_rows; idx++)
        {
            hsize_t size = input_vector[idx].size();

            varlen_spec.at(idx).len = size;
            varlen_spec.at(idx).p = (void *)&input_vector[idx].front();
        }

        return varlen_spec;
    }

    template<typename T>
    H5::FloatType get_hdf5_type() {
        H5::FloatType item_type(H5::PredType::IEEE_F32LE);

        return item_type;
    }

    template <typename T>
    void write_var_length_array(H5::H5File file,const std::string& dataset_name,std::vector<std::vector<T>>& input_data) {

        const hsize_t n_dims = 1;
        const hsize_t n_rows = input_data.size();
    
        // this structure stores length of each varlen row and a pointer to
        // the actual data
        std::vector<hvl_t> varlen_spec = make_varlen(n_rows,input_data);

        H5::FloatType item_type = get_hdf5_type<T>();

        H5::VarLenType file_type(item_type);

        H5::DataSpace dataspace(n_dims, &n_rows);

        H5::DataSet dataset = file.createDataSet(dataset_name, file_type, dataspace);

        // dtype of the generated data
        H5::VarLenType mem_type(item_type);

        dataset.write(&varlen_spec.front(), mem_type);

    }

    std::vector<int> frame;
    std::vector<std::vector<uint16_t>> x_pos;
    std::vector<std::vector<uint16_t>> y_pos;
    std::vector<std::vector<float>> prob;
    int x_dim;
    int y_dim;
    std::vector<uint8_t> prediction_frame;
    // https://github.com/BlueBrain/HighFive/pull/578
    // Cannot have file member
};

template<> inline H5::FloatType save_structure::get_hdf5_type<float>() {
    H5::FloatType item_type(H5::PredType::IEEE_F32LE);

    return item_type;
};
template<> inline H5::FloatType save_structure::get_hdf5_type<int>() {
    H5::FloatType item_type(H5::PredType::STD_I32LE);

    return item_type;
};
template<> inline H5::FloatType save_structure::get_hdf5_type<uint16_t>() {
    H5::FloatType item_type(H5::PredType::STD_U16LE);

    return item_type;
}
# Hourglass

## Overview

This is a C++ and LibTorch implementation of a stacked hourglass neural network for pose detection. This model was first described in Newell et al 2016:  
https://arxiv.org/abs/1603.06937  
  
I have previously made a Julia implementation of this type of network, which can be found here:  
https://github.com/paulmthompson/StackedHourglass.jl
  
An excellent PyTorch implemention by another author can be found here:  
https://github.com/princeton-vl/pytorch_stacked_hourglass

## Install
  
Libtorch (https://pytorch.org/get-started/locally/)  
  
I use VCPKG to install the other dependencies which are  
* nlohmann_json (https://github.com/nlohmann/json)  
* cxxopts (https://github.com/jarro2783/cxxopts)  
* OpenCV (https://opencv.org/) 
* HighFive (https://github.com/BlueBrain/HighFive)
* hdf5[cpp] (http://davis.lbl.gov/Manuals/HDF5-1.8.7/cpplus_RM/index.html) 
  
## Usage

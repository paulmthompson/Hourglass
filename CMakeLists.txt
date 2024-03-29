cmake_minimum_required(VERSION 3.21)
project(Hourglass)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#[[
Manifest mode from VCPKG will make all dependancies specified in vcpkg.json be associated and installed
with this particular project
]]
set(VCPKG_MANIFEST_MODE ON)

#[[
We will get external package ffmpeg_wrapper using fetchcontent
]]
add_subdirectory(external)

#[[
I am using vcpkg manifest to install the following packages. Libtorch (where I am not using VCPKG) is going have some shared dependancies,
specifically at least protobuf, and will error if it imports first from outside of vcpkg. Consequently, it is best to use find_package with all
of the vcpkg dependancies FIRST
]]
find_package(nlohmann_json CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(HighFive CONFIG REQUIRED)
find_package(hdf5 CONFIG REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)

add_executable(Hourglass 
  hourglass.cpp 
  include/hourglass.hpp 
  include/residual.hpp 
  include/training.hpp
  include/prediction.hpp
  include/saveload.hpp
  include/subpixel.hpp
  )

add_subdirectory(dataloader)

target_include_directories(Hourglass PRIVATE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dataloader>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

if( OpenCV_FOUND )
  # Additional Include Directories
  target_include_directories(Hourglass PRIVATE 
        "<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>" 
        )

  # Additional Library Directories
  target_link_directories(Hourglass PRIVATE 
        "<BUILD_INTERFACE:${OpenCV_LIB_DIR}>" 
        )

  # Additional Dependencies
  target_link_libraries(Hourglass PRIVATE ${OpenCV_LIBS} )
endif()

target_link_libraries(Hourglass PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(Hourglass PRIVATE cxxopts::cxxopts)
target_link_libraries(Hourglass PRIVATE PNG::PNG)
target_link_libraries(Hourglass PRIVATE ffmpeg_wrapper::ffmpeg_wrapper)
target_link_libraries(Hourglass PRIVATE HighFive)
target_link_libraries(Hourglass PRIVATE hdf5::hdf5-shared hdf5::hdf5_hl-shared hdf5::hdf5_cpp-shared hdf5::hdf5_hl_cpp-shared)

target_link_libraries(Hourglass PRIVATE dataloader)

target_link_libraries(Hourglass PRIVATE "${TORCH_LIBRARIES}")


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Hourglass
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Hourglass>)
endif (MSVC)
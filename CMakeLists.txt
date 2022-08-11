cmake_minimum_required(VERSION 3.21)
project(Hourglass)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(ffmpeg_wrapper CONFIG REQUIRED)

#Image Loading
find_package(PNG REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(Hourglass hourglass.cpp 
  include/hourglass.hpp 
  include/residual.hpp 
  include/dataload.hpp 
  include/training.hpp
  include/augmentation.hpp
  include/prediction.hpp
  )

if( OpenCV_FOUND )
  # Additional Include Directories
  target_include_directories(Hourglass PRIVATE ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  target_link_directories(Hourglass PRIVATE ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  target_link_libraries(Hourglass PRIVATE ${OpenCV_LIBS} )
endif()

target_link_libraries(Hourglass PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(Hourglass PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(Hourglass PRIVATE cxxopts::cxxopts)
target_link_libraries(Hourglass PRIVATE PNG::PNG)
target_link_libraries(Hourglass PRIVATE ffmpeg_wrapper::ffmpeg_wrapper)

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
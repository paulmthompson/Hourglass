cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#This is necessary to work on windows. A better way would be to define only the symbols that need to be exported
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(dataloader SHARED 
                        augmentation.hpp 
                        dataload.hpp 
                        dataload.cpp)


target_include_directories(dataloader PUBLIC 
                                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")

if( OpenCV_FOUND )
  # Additional Include Directories
  target_include_directories(dataloader PRIVATE 
        "<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>" 
        )

  # Additional Library Directories
  target_link_directories(dataloader PRIVATE 
        "<BUILD_INTERFACE:${OpenCV_LIB_DIR}>" 
        )

  # Additional Dependencies
  target_link_libraries(dataloader PRIVATE ${OpenCV_LIBS} )
endif()

target_link_libraries(dataloader PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(dataloader PRIVATE nlohmann_json::nlohmann_json)
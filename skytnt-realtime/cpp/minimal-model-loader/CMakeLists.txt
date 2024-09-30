# CMake minimum version
cmake_minimum_required(VERSION 3.15)

# Set project name and specify C++ as the language
project(MyApp LANGUAGES CXX OBJCXX)

# Specify the C++ standard for C++ files only
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Objective-C++ and ARC (Automatic Reference Counting)
set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")

# List source files - this 
#set(SOURCES main.mm newmodel.m)

# Add the executable
#add_executable(MyApp ${SOURCES})

add_executable(MyApp main.mm newmodel.m)

# Link required frameworks (Foundation and CoreML)
find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(COREML_FRAMEWORK CoreML)

target_link_libraries(MyApp ${FOUNDATION_FRAMEWORK} ${COREML_FRAMEWORK})

# Specify the use of clang++ if necessary
set(CMAKE_CXX_COMPILER "clang++")

# Apply C++17 standard only for C++ files (not for Objective-C++)
set_source_files_properties(main.mm newmodel.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")


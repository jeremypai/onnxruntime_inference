cmake_minimum_required(VERSION 3.21)

project(ONNXRuntimeTest LANGUAGES CXX C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()
include(GoogleTest)

add_subdirectory(thirdparty)

add_executable(${PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/onnxruntime_inference_test.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE ONNXRuntimeExecutor
    PRIVATE ImageIO
    PRIVATE GTest::gtest_main)

gtest_discover_tests(${PROJECT_NAME})

add_subdirectory(c_api/image_manipulate)
add_subdirectory(c_api/onnxruntime_executor)
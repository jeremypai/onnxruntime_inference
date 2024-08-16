project(ImageIOTest LANGUAGES CXX C)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/image_manipulate_test.cpp)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME}
    PRIVATE ImageIO
    PRIVATE GTest::gtest_main)

gtest_discover_tests(${PROJECT_NAME})
# ONNX Runtime
set(onnxruntime_version 1.18.0)

if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-linux-x64-gpu-${onnxruntime_version})
    file (ARCHIVE_EXTRACT 
        INPUT ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-cuda12-${onnxruntime_version}.tgz
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

find_path (onnxruntime_include
    NAMES onnxruntime_c_api.h
    PATHS ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-linux-x64-gpu-${onnxruntime_version}/include
    REQUIRED)

find_library(onnxruntime_lib 
    NAMES onnxruntime
    PATHS ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-linux-x64-gpu-${onnxruntime_version}/lib
    REQUIRED)

add_library(onnxruntime INTERFACE)

target_include_directories(onnxruntime INTERFACE ${onnxruntime_include})

target_link_libraries(onnxruntime INTERFACE ${onnxruntime_lib})

# Googletest
set(googletest_version 1.15.2)

if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/googletest-${googletest_version})
    file (ARCHIVE_EXTRACT
        INPUT ${CMAKE_CURRENT_SOURCE_DIR}/googletest-${googletest_version}.tar.gz
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(BUILD_GMOCK OFF)
set(INSTALL_GTEST OFF)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-${googletest_version} 
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-${googletest_version}-build)
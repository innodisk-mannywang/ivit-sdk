# ============================================================================
# Dependencies.cmake
# 依賴庫管理 - 支援系統安裝或打包模式
# ============================================================================

include(FetchContent)
include(ExternalProject)

# 依賴庫目錄
set(IVIT_DEPS_DIR "${CMAKE_SOURCE_DIR}/deps" CACHE PATH "Dependencies directory")
set(IVIT_DEPS_INSTALL_DIR "${IVIT_DEPS_DIR}/install" CACHE PATH "Dependencies install directory")

# 打包模式選項
option(IVIT_BUNDLE_DEPS "Bundle dependencies into SDK" OFF)
option(IVIT_DOWNLOAD_DEPS "Download pre-built dependencies" OFF)

# ============================================================================
# RPATH 設定 - 讓 SDK 能找到打包的依賴庫
# ============================================================================
if(IVIT_BUNDLE_DEPS)
    # 設定 RPATH 為相對路徑
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;$ORIGIN/../deps/lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    # 同時設定 build 時的 RPATH
    set(CMAKE_BUILD_RPATH "$ORIGIN;${IVIT_DEPS_INSTALL_DIR}/lib")
endif()

# ============================================================================
# OpenVINO 依賴
# ============================================================================
macro(setup_openvino)
    if(IVIT_BUNDLE_DEPS AND EXISTS "${IVIT_DEPS_INSTALL_DIR}/runtime/lib")
        # 使用打包的 OpenVINO
        message(STATUS "Using bundled OpenVINO")

        set(OpenVINO_DIR "${IVIT_DEPS_INSTALL_DIR}/runtime/cmake")
        find_package(OpenVINO REQUIRED COMPONENTS Runtime)

        # 標記需要打包的庫
        file(GLOB OPENVINO_LIBS "${IVIT_DEPS_INSTALL_DIR}/runtime/lib/*.so*")
        list(APPEND IVIT_BUNDLE_LIBRARIES ${OPENVINO_LIBS})

    elseif(IVIT_DOWNLOAD_DEPS)
        # 下載預編譯的 OpenVINO
        message(STATUS "Downloading OpenVINO...")

        set(OV_VERSION "2024.0.0")
        set(OV_URL "https://storage.openvinotoolkit.org/repositories/openvino/packages/${OV_VERSION}/linux/l_openvino_toolkit_ubuntu22_${OV_VERSION}.tgz")

        FetchContent_Declare(
            openvino_pkg
            URL ${OV_URL}
            SOURCE_DIR ${IVIT_DEPS_DIR}/openvino
        )
        FetchContent_MakeAvailable(openvino_pkg)

        set(OpenVINO_DIR "${IVIT_DEPS_DIR}/openvino/runtime/cmake")
        find_package(OpenVINO REQUIRED COMPONENTS Runtime)

    else()
        # 使用系統安裝的 OpenVINO
        find_package(OpenVINO QUIET COMPONENTS Runtime)
    endif()
endmacro()

# ============================================================================
# ONNX Runtime 依賴
# ============================================================================
macro(setup_onnxruntime)
    if(IVIT_BUNDLE_DEPS AND EXISTS "${IVIT_DEPS_INSTALL_DIR}/onnxruntime")
        # 使用打包的 ONNX Runtime
        message(STATUS "Using bundled ONNX Runtime")

        set(ONNXRUNTIME_ROOT "${IVIT_DEPS_INSTALL_DIR}/onnxruntime")
        set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
        set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

        # 標記需要打包的庫
        file(GLOB ORT_LIBS "${ONNXRUNTIME_ROOT}/lib/*.so*")
        list(APPEND IVIT_BUNDLE_LIBRARIES ${ORT_LIBS})

    elseif(IVIT_DOWNLOAD_DEPS)
        # 下載預編譯的 ONNX Runtime
        message(STATUS "Downloading ONNX Runtime...")

        set(ORT_VERSION "1.17.0")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(ORT_ARCH "aarch64")
        else()
            set(ORT_ARCH "x64")
        endif()

        # GPU 版本
        if(CUDA_FOUND)
            set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-gpu-${ORT_VERSION}.tgz")
        else()
            set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz")
        endif()

        FetchContent_Declare(
            onnxruntime_pkg
            URL ${ORT_URL}
            SOURCE_DIR ${IVIT_DEPS_DIR}/onnxruntime
        )
        FetchContent_MakeAvailable(onnxruntime_pkg)

        set(ONNXRUNTIME_ROOT "${IVIT_DEPS_DIR}/onnxruntime")
        set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
        set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

    else()
        # 使用系統安裝的 ONNX Runtime
        find_package(onnxruntime QUIET)
        if(NOT onnxruntime_FOUND)
            find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
                HINTS ${ONNXRUNTIME_ROOT} /usr/local /usr
                PATH_SUFFIXES include onnxruntime/include)
            find_library(ONNXRUNTIME_LIBRARY onnxruntime
                HINTS ${ONNXRUNTIME_ROOT} /usr/local /usr
                PATH_SUFFIXES lib lib64)
        endif()
    endif()
endmacro()

# ============================================================================
# TensorRT 依賴 (僅支援動態連結)
# ============================================================================
macro(setup_tensorrt)
    if(IVIT_BUNDLE_DEPS AND EXISTS "${IVIT_DEPS_INSTALL_DIR}/tensorrt")
        # 使用打包的 TensorRT
        message(STATUS "Using bundled TensorRT")

        set(TENSORRT_ROOT "${IVIT_DEPS_INSTALL_DIR}/tensorrt")
        set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
        find_library(TENSORRT_LIBRARY nvinfer
            PATHS "${TENSORRT_ROOT}/lib"
            NO_DEFAULT_PATH)
        find_library(NVONNXPARSER_LIBRARY nvonnxparser
            PATHS "${TENSORRT_ROOT}/lib"
            NO_DEFAULT_PATH)

        # 標記需要打包的庫
        file(GLOB TRT_LIBS "${TENSORRT_ROOT}/lib/*.so*")
        list(APPEND IVIT_BUNDLE_LIBRARIES ${TRT_LIBS})

    else()
        # 使用系統安裝的 TensorRT
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/local/tensorrt /usr
            PATH_SUFFIXES include)
        find_library(TENSORRT_LIBRARY nvinfer
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/local/tensorrt /usr
            PATH_SUFFIXES lib lib64)
        find_library(NVONNXPARSER_LIBRARY nvonnxparser
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/local/tensorrt /usr
            PATH_SUFFIXES lib lib64)
    endif()
endmacro()

# ============================================================================
# CUDA Runtime (僅支援動態連結)
# ============================================================================
macro(setup_cuda)
    if(IVIT_BUNDLE_DEPS AND EXISTS "${IVIT_DEPS_INSTALL_DIR}/cuda")
        message(STATUS "Using bundled CUDA Runtime")

        set(CUDA_TOOLKIT_ROOT_DIR "${IVIT_DEPS_INSTALL_DIR}/cuda")
        set(CUDAToolkit_ROOT "${IVIT_DEPS_INSTALL_DIR}/cuda")

        # 標記需要打包的庫 (只打包 runtime，不打包開發工具)
        file(GLOB CUDA_RUNTIME_LIBS
            "${IVIT_DEPS_INSTALL_DIR}/cuda/lib64/libcudart.so*"
            "${IVIT_DEPS_INSTALL_DIR}/cuda/lib64/libcublas.so*"
            "${IVIT_DEPS_INSTALL_DIR}/cuda/lib64/libcublasLt.so*"
        )
        list(APPEND IVIT_BUNDLE_LIBRARIES ${CUDA_RUNTIME_LIBS})
    endif()

    find_package(CUDAToolkit QUIET)
endmacro()

# ============================================================================
# 安裝打包的依賴庫
# ============================================================================
function(install_bundled_dependencies)
    if(NOT IVIT_BUNDLE_DEPS)
        return()
    endif()

    # 建立 deps/lib 目錄
    install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/deps/lib)

    # 複製所有依賴庫
    foreach(lib ${IVIT_BUNDLE_LIBRARIES})
        if(EXISTS ${lib})
            # 處理 symlink
            get_filename_component(lib_name ${lib} NAME)
            get_filename_component(lib_realpath ${lib} REALPATH)

            install(FILES ${lib_realpath}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/deps/lib
                RENAME ${lib_name})
        endif()
    endforeach()
endfunction()

# ============================================================================
# 建立依賴庫載入腳本
# ============================================================================
function(create_env_script)
    if(NOT IVIT_BUNDLE_DEPS)
        return()
    endif()

    # 建立 setup_env.sh
    file(WRITE "${CMAKE_BINARY_DIR}/setup_env.sh" [[
#!/bin/bash
# iVIT-SDK 環境設定腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(dirname "$SCRIPT_DIR")"

# 設定 LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${SDK_ROOT}/lib:${SDK_ROOT}/deps/lib:${LD_LIBRARY_PATH}"

# 設定 Python 路徑
export PYTHONPATH="${SDK_ROOT}/python:${PYTHONPATH}"

echo "iVIT-SDK environment configured."
echo "  SDK_ROOT: ${SDK_ROOT}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
]])

    install(PROGRAMS "${CMAKE_BINARY_DIR}/setup_env.sh"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endfunction()

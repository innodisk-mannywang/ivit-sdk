# iVIT-SDK Installation Guide

## Prerequisites

- Ubuntu 20.04 / 22.04 (x86_64 or aarch64)
- OpenCV 4.x (`sudo apt install libopencv-dev`)
- Platform-specific runtime:
  - **NVIDIA**: CUDA Toolkit + TensorRT
  - **Intel**: OpenVINO Runtime 2024+

## Quick Start

```bash
# 1. Extract the tarball
tar -xzf ivit-sdk-*.tar.gz
cd ivit-sdk-*

# 2. Set up environment
source bin/setup_env.sh

# 3. Verify installation
python3 -c "import ivit; print(ivit.__version__)"
```

## Directory Structure

```
ivit-sdk-<version>-<platform>/
├── lib/libivit.so              # Core shared library
├── lib/cmake/ivit/             # CMake find_package support
├── include/ivit/               # C++ headers
├── deps/lib/                   # Bundled runtime dependencies
├── python/                     # Python wheel
├── bin/setup_env.sh            # Environment setup script
├── examples/                   # C++ and Python examples
└── INSTALL.md                  # This file
```

## C++ Usage

```bash
# Set environment first
source bin/setup_env.sh

# Compile with CMake
cmake -Divit_DIR=<sdk-path>/lib/cmake/ivit ..
```

Or compile directly:

```bash
g++ -I<sdk-path>/include -L<sdk-path>/lib -livit my_app.cpp -o my_app
```

## Python Usage

```bash
# Install the wheel
pip install python/ivit_sdk-*.whl

# Or use PYTHONPATH (already set by setup_env.sh)
python3 -c "import ivit"
```

## Troubleshooting

**Library not found errors**: Make sure you've run `source bin/setup_env.sh` or added `lib/` and `deps/lib/` to `LD_LIBRARY_PATH`.

**GPU not detected**: Verify your GPU driver and runtime (CUDA/OpenVINO) are correctly installed.

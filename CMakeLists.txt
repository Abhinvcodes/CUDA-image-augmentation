# Require at least version 3.18, which introduced native CUDA support
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Define the project and explicitly enable C and CUDA languages
project(CudaImageAugmentation LANGUAGES C CUDA)

# Force the C standard (C99 is standard, but you can change to C11 if needed)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optional: Uncomment and set this to your specific GPU architecture to optimize compiling
# (e.g., 75 for Turing/RTX 20-series, 86 for Ampere/RTX 30-series, 89 for Ada/RTX 40-series)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Tell CMake to compile both main.cu and kernels.cu into the augment binary
add_executable(augment src/main.cu src/kernels.cu)

# Tell the compiler where to look for header files (like stb_image.h)
target_include_directories(augment PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
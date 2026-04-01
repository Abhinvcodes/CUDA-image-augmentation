# CUDA Image Augmentation Pipeline

A from-scratch, natively compiled CUDA pipeline for high-performance image augmentation. 

This project is designed to accelerate the generation of machine learning training data by offloading heavy image processing operations (scaling, blurring, sharpening, and rotation) to the GPU. It utilizes a custom C/CUDA architecture and a lightweight dependency stack to minimize overhead and maximize parallel thread efficiency.

## Tech Stack & Dependencies
* **Language:** C / CUDA
* **Build System:** CMake (3.18+)
* **Compiler:** NVCC (NVIDIA CUDA Compiler) & GCC
* **Image I/O:** `stb_image` (included as single-header libraries)

## Project Structure
```text
├── include/                 # Third-party single-header libraries
│   ├── stb_image.h          # Decodes JPEGs/PNGs into raw pixel arrays
│   └── stb_image_write.h    # Encodes raw pixel arrays back to disk
├── src/
│   └── main.cu              # Core host/device code and kernel implementations
├── .gitignore               # Ignores build artifacts and generated outputs
├── CMakeLists.txt           # Build system configuration
└── README.md                # Project documentation
```
## Prerequisites
* **CUDA Toolkit**
* **CMake**
* **Base Development Tools**

*(On Arch Linux, these can be installed via sudo pacman -S cuda cmake base-devel)*

## Build and Execution Instructions
1. Prepare the input
Place a test image named input.jpg directly into the root directory of this project (alongside the CMakeLists.txt file).

2. Compile the Project
```text
# Create a dedicated build directory
mkdir build
cd build

# Generate the Makefiles using CMake
cmake ..

# Compile the CUDA and C code
make
```

3. Run the Pipeline
```text
./augment
```

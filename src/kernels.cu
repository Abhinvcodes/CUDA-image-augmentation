#include <stdio.h>
#include "../include/kernels.h"

// -------------------------------------------------------------------------
// COPY KERNEL
// -------------------------------------------------------------------------

__global__ void copyKernel(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int pixel_index = (y * width + x) * channels;
        for (int c = 0; c < channels; ++c)
        {
            output[pixel_index + c] = input[pixel_index + c];
        }
    }
}

// 2. The Wrapper Function (Publicly accessible)
void runCopyKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels)
{
    // Configure execution
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)...\n", grid.x, grid.y, block.x, block.y);

    // Launch the kernel
    copyKernel<<<grid, block>>>(d_input, d_output, width, height, channels);

    // Check for launch errors silently
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
    }
}
// -------------------------------------------------------------------------
// BLUR KERNEL
// -------------------------------------------------------------------------

__global__ void blurKernel(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x < width && y < height)
    {

        // We will calculate the blur for each color channel separately (R, G, B)
        for (int c = 0; c < channels; ++c)
        {
            int color_sum = 0;
            int pixels_counted = 0;

            // Loop over a 3x3 grid around the current pixel
            for (int blur_y = -1; blur_y <= 1; ++blur_y)
            {
                for (int blur_x = -1; blur_x <= 1; ++blur_x)
                {

                    int neighbor_x = x + blur_x;
                    int neighbor_y = y + blur_y;

                    // Ensure the neighbor is actually inside the image bounds
                    if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height)
                    {
                        int neighbor_index = (neighbor_y * width + neighbor_x) * channels + c;
                        color_sum += input[neighbor_index];
                        pixels_counted++;
                    }
                }
            }

            // Calculate the average color and assign it to the output
            int current_pixel_index = (y * width + x) * channels + c;
            output[current_pixel_index] = (unsigned char)(color_sum / pixels_counted);
        }
    }
}

// The Wrapper
void runBlurKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    printf("Applying 3x3 Box Blur...\n");
    blurKernel<<<grid, block>>>(d_input, d_output, width, height, channels);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
    }
}
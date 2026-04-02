#include <stdio.h>
#include "../include/kernels.h"
// A macro to check and CLEAR the last kernel error
#define CHECK_KERNEL(name)                                                      \
    {                                                                           \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess)                                                 \
        {                                                                       \
            printf(">> CUDA ERROR in %s: %s\n", name, cudaGetErrorString(err)); \
        }                                                                       \
    }

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
    CHECK_KERNEL("Copy");
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

    CHECK_KERNEL("Blur");
}

__global__ void sharpenKernel(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // A standard 3x3 Sharpening matrix
        int filter[3][3] = {
            {0, -1, 0},
            {-1, 5, -1},
            {0, -1, 0}};

        for (int c = 0; c < channels; ++c)
        {
            int color_val = 0;

            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    int nx = x + kx;
                    int ny = y + ky;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                    {
                        int index = (ny * width + nx) * channels + c;
                        color_val += input[index] * filter[ky + 1][kx + 1];
                    }
                }
            }

            // Clamp the values to valid RGB ranges
            if (color_val < 0)
                color_val = 0;
            if (color_val > 255)
                color_val = 255;

            output[(y * width + x) * channels + c] = (unsigned char)color_val;
        }
    }
}

void runSharpenKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    sharpenKernel<<<grid, block>>>(d_input, d_output, width, height, channels);

    CHECK_KERNEL("Sharpen");
}

__global__ void rotateKernel(unsigned char *input, unsigned char *output, int width, int height, int channels, float angle_rad)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float cx = width / 2.0f;
        float cy = height / 2.0f;

        float s = sin(angle_rad);
        float c_cos = cos(angle_rad);

        // Inverse rotation map
        int src_x = (int)((x - cx) * c_cos + (y - cy) * s + cx);
        int src_y = (int)(-(x - cx) * s + (y - cy) * c_cos + cy);

        int out_idx = (y * width + x) * channels;

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height)
        {
            int in_idx = (src_y * width + src_x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                output[out_idx + c] = input[in_idx + c];
            }
        }
        else
        {
            // If the source pixel is outside the original image, paint it black
            for (int c = 0; c < channels; ++c)
            {
                output[out_idx + c] = 0;
            }
        }
    }
}

void runRotateKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Hardcoding a 45 degree rotation (0.785398 radians) for now
    float angle = 0.785398f;
    rotateKernel<<<grid, block>>>(d_input, d_output, width, height, channels, angle);

    CHECK_KERNEL("Rotate");
}

__global__ void scaleKernel(unsigned char *input, unsigned char *output, int width, int height, int channels, float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float cx = width / 2.0f;
        float cy = height / 2.0f;

        // Inverse scaling map
        int src_x = (int)((x - cx) / scale + cx);
        int src_y = (int)((y - cy) / scale + cy);

        int out_idx = (y * width + x) * channels;

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height)
        {
            int in_idx = (src_y * width + src_x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                output[out_idx + c] = input[in_idx + c];
            }
        }
        else
        {
            for (int c = 0; c < channels; ++c)
            {
                output[out_idx + c] = 0;
            }
        }
    }
}

void runScaleKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Hardcoding a 1.5x zoom
    float scale = 1.5f;
    scaleKernel<<<grid, block>>>(d_input, d_output, width, height, channels, scale);

    CHECK_KERNEL("Scale");
}
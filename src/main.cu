#include <stdio.h>
#include <stdlib.h>

// 1. Define stb macros and include headers
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

// -------------------------------------------------------------------------
// DEVICE CODE (Runs on the GPU)
// -------------------------------------------------------------------------

__global__ void copyKernel(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    // 1. Calculate the global X and Y coordinates for this specific thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Boundary check: Ensure the thread doesn't read/write outside the image
    if (x < width && y < height)
    {

        // 3. Flatten the 2D (x, y) coordinate into a 1D array index
        int pixel_index = (y * width + x) * channels;

        // 4. Copy all color channels for this specific pixel
        for (int c = 0; c < channels; ++c)
        {
            output[pixel_index + c] = input[pixel_index + c];
        }
    }
}

// -------------------------------------------------------------------------
// HOST CODE (Runs on the CPU)
// -------------------------------------------------------------------------

int main()
{
    // --- 1. Load the Image (Host) ---
    int width, height, channels;
    const char *input_filename = "../input.jpg";

    printf("Loading image: %s\n", input_filename);

    unsigned char *h_input = stbi_load(input_filename, &width, &height, &channels, 0);
    if (h_input == NULL)
    {
        printf("Error: Could not load image %s\n", input_filename);
        return -1;
    }
    printf("Image loaded: %d x %d pixels, %d channels\n", width, height, channels);

    size_t img_size = width * height * channels * sizeof(unsigned char);

    // --- 2. Allocate GPU Memory (Device) ---
    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, img_size);
    cudaMalloc((void **)&d_output, img_size);

    // --- 3. Copy Data: CPU -> GPU ---
    printf("Transferring data to GPU...\n");
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    // --- 4. Configure Kernel Execution ---
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    printf("Launching Kernel with Grid(%d, %d) and Block(%d, %d)...\n", grid.x, grid.y, block.x, block.y);

    // --- 5. Launch Kernel ---
    copyKernel<<<grid, block>>>(d_input, d_output, width, height, channels);

    cudaDeviceSynchronize();

    // --- 6. Copy Data: GPU -> CPU ---
    printf("Transferring data back to CPU...\n");
    unsigned char *h_output = (unsigned char *)malloc(img_size);
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    // --- 7. Save the Image (Host) ---
    const char *output_filename = "../output_copy.png";
    printf("Saving image to: %s\n", output_filename);

    stbi_write_png(output_filename, width, height, channels, h_output, width * channels);

    // --- 8. Cleanup ---
    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);
    free(h_output);

    printf("Done!\n");
    return 0;
}
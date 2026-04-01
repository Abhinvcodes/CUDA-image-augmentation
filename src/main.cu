// src/main.cu
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

// Include your new custom header
#include "../include/kernels.h"

int main()
{
    int width, height, channels;
    const char *input_filename = "../input.jpg";

    unsigned char *h_input = stbi_load(input_filename, &width, &height, &channels, 0);
    size_t img_size = width * height * channels * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, img_size);
    cudaMalloc((void **)&d_output, img_size);

    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    // --- MAGICAL CLEAN KERNEL CALL ---
    // runCopyKernel(d_input, d_output, width, height, channels);
    // cudaDeviceSynchronize();

    runBlurKernel(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    unsigned char *h_output = (unsigned char *)malloc(img_size);
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    const char *output_filename = "../output_copy.png";
    stbi_write_png(output_filename, width, height, channels, h_output, width * channels);

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);
    free(h_output);

    return 0;
}
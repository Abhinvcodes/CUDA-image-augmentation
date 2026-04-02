#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include "../include/kernels.h"

// Helper function to swap pointers for our Ping-Pong buffer
void swapPointers(unsigned char **a, unsigned char **b)
{
    unsigned char *temp = *a;
    *a = *b;
    *b = temp;
}

int main()
{
    int width, height, channels;
    const char *input_filename = "../input.png";

    unsigned char *h_input = stbi_load(input_filename, &width, &height, &channels, 0);
    size_t img_size = width * height * channels * sizeof(unsigned char);

    // Open manifest file in append mode
    FILE *manifest = fopen("../data/manifest.csv", "a");
    if (manifest == NULL)
    {
        printf("Error: Could not open manifest.csv\n");
        return -1;
    }

    // Allocate Ping-Pong GPU Buffers
    unsigned char *d_buffer_A, *d_buffer_B;
    cudaMalloc((void **)&d_buffer_A, img_size);
    cudaMalloc((void **)&d_buffer_B, img_size);

    unsigned char *h_output = (unsigned char *)malloc(img_size);

    printf("Starting Power Set Augmentation (32 Combinations)...\n");

    // Loop from 0 to 31 (00000 to 11111 in binary)
    for (int i = 0; i < 16; ++i)
    {

        // 1. Reset Buffer A to the original image for each new combination
        cudaMemcpy(d_buffer_A, h_input, img_size, cudaMemcpyHostToDevice);

        // Pointers to keep track of active source and destination
        unsigned char *d_src = d_buffer_A;
        unsigned char *d_dst = d_buffer_B;

        // String to build our filename and manifest log
        char applied_augs[256] = "";
        char filename[256] = "";

        // 2. The Bitmask Checks
        // We use bitwise AND (&) to check if a specific bit is a 1 or 0

        // Bit 0: Blur
        if ((i & (1 << 0)) != 0)
        {
            runBlurKernel(d_src, d_dst, width, height, channels);
            strcat(applied_augs, "blur_");
            swapPointers(&d_src, &d_dst); // Output is now the input for the next step
        }

        // Bit 1: Rotate
        if ((i & (1 << 1)) != 0)
        {
            runRotateKernel(d_src, d_dst, width, height, channels);
            strcat(applied_augs, "rot_");
            swapPointers(&d_src, &d_dst);
        }

        // Bit 2: Sharpen
        if ((i & (1 << 2)) != 0)
        {
            runSharpenKernel(d_src, d_dst, width, height, channels);
            strcat(applied_augs, "sharp_");
            swapPointers(&d_src, &d_dst);
        }

        // Bit 3: Scale
        if ((i & (1 << 3)) != 0)
        {
            runScaleKernel(d_src, d_dst, width, height, channels);
            strcat(applied_augs, "scale_");
            swapPointers(&d_src, &d_dst);
        }

        // Bit 4: Resize
        /*
        if ((i & (1 << 4)) != 0)
        {
            runResizeKernel(d_src, d_dst, width, height, channels);
            strcat(applied_augs, "resz_");
            swapPointers(&d_src, &d_dst);
        }*/

        // Ensure the GPU has finished all kernels for this combination
        cudaDeviceSynchronize();

        // 3. Save the Result
        // If no augs were applied, name it 'original'
        if (i == 0)
        {
            strcpy(applied_augs, "original");
        }

        // Ensure no trailing underscore
        if (strlen(applied_augs) > 0 && applied_augs[strlen(applied_augs) - 1] == '_')
        {
            applied_augs[strlen(applied_augs) - 1] = '\0';
        }

        // Change the path to the new data/augmented/ directory
        snprintf(filename, sizeof(filename), "../data/augmented/output_%02d_%s.png", i, applied_augs);

        // Copy the FINAL d_src back to the CPU
        cudaMemcpy(h_output, d_src, img_size, cudaMemcpyDeviceToHost);
        stbi_write_png(filename, width, height, channels, h_output, width * channels);

        // 4. Log to CSV
        fprintf(manifest, "%s, %s\n", filename, applied_augs);
        printf("Generated: %s\n", filename);
    }

    // Cleanup
    fclose(manifest);
    cudaFree(d_buffer_A);
    cudaFree(d_buffer_B);
    stbi_image_free(h_input);
    free(h_output);

    printf("Done!\n");
    return 0;
}
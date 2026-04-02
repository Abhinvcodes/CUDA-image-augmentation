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

    // Ensure you are targeting your PNG file!
    const char *input_filename = "../input.png";

    unsigned char *h_input = stbi_load(input_filename, &width, &height, &channels, 0);

    // Safety check: Ensure the image actually loaded
    if (h_input == NULL)
    {
        printf("CRITICAL ERROR: Could not load image from %s\n", input_filename);
        printf("Reason: %s\n", stbi_failure_reason());
        return -1;
    }

    // Open manifest file in append mode
    FILE *manifest = fopen("../data/manifest.csv", "a");
    if (manifest == NULL)
    {
        printf("Error: Could not open manifest.csv. Did you create the data/ folder?\n");
        return -1;
    }

    // --- 1. DYNAMIC MEMORY BOUNDS ---
    // Instead of using the 220x220 size, we allocate for the absolute maximum
    // size this pipeline could produce so the GPU doesn't crash on expansion.
    int max_w = 500;
    int max_h = 500;
    size_t max_img_size = max_w * max_h * channels * sizeof(unsigned char);

    // Allocate Ping-Pong GPU Buffers to hold up to 500x500
    unsigned char *d_buffer_A, *d_buffer_B;
    cudaMalloc((void **)&d_buffer_A, max_img_size);
    cudaMalloc((void **)&d_buffer_B, max_img_size);

    // Host output buffer also needs max capacity
    unsigned char *h_output = (unsigned char *)malloc(max_img_size);

    printf("Starting Power Set Augmentation (32 Combinations)...\n");

    // --- 2. THE PIPELINE LOOP (0 to 31) ---
    for (int i = 0; i < 32; ++i)
    {

        // At the start of every combination, explicitly reset to the original dimensions
        int current_w = width;
        int current_h = height;

        // Only copy the bytes needed for the original 220x220 image into the start of Buffer A
        size_t original_size = current_w * current_h * channels * sizeof(unsigned char);
        cudaMemcpy(d_buffer_A, h_input, original_size, cudaMemcpyHostToDevice);

        // Reset the active pointers
        unsigned char *d_src = d_buffer_A;
        unsigned char *d_dst = d_buffer_B;

        char applied_augs[256] = "";
        char filename[256] = "";

        // Bit 0: Blur
        if ((i & (1 << 0)) != 0)
        {
            runBlurKernel(d_src, d_dst, current_w, current_h, channels);
            strcat(applied_augs, "blur_");
            swapPointers(&d_src, &d_dst);
        }

        // Bit 1: Rotate
        if ((i & (1 << 1)) != 0)
        {
            runRotateKernel(d_src, d_dst, current_w, current_h, channels);
            strcat(applied_augs, "rot_");
            swapPointers(&d_src, &d_dst);
        }

        // Bit 2: Sharpen
        if ((i & (1 << 2)) != 0)
        {
            runSharpenKernel(d_src, d_dst, current_w, current_h, channels);
            strcat(applied_augs, "sharp_");
            swapPointers(&d_src, &d_dst);
        }

        // Bit 3: Scale (Zooming in 1.5x, but keeping canvas size the same)
        if ((i & (1 << 3)) != 0)
        {
            runScaleKernel(d_src, d_dst, current_w, current_h, channels);
            strcat(applied_augs, "scale_");
            swapPointers(&d_src, &d_dst);
        }

        // Bit 4: RESIZE (Expanding the canvas to 500x500)
        // Since we evaluate top-to-bottom, this happens last. This is highly optimized!
        // We only do math on 48,400 pixels for blur/rotate, and stretch it at the very end.
        if ((i & (1 << 4)) != 0)
        {
            runResizeKernel(d_src, d_dst, current_w, current_h, max_w, max_h, channels);

            // --- CRITICAL STEP ---
            // Update the state trackers so the saving logic knows the image just got bigger!
            current_w = max_w;
            current_h = max_h;

            strcat(applied_augs, "resz_");
            swapPointers(&d_src, &d_dst);
        }

        // Ensure all kernels for this combination are finished before saving
        cudaDeviceSynchronize();

        // --- 3. SAVING THE RESULT ---
        // Clean up the string name
        if (i == 0)
        {
            strcpy(applied_augs, "original");
        }
        else if (strlen(applied_augs) > 0 && applied_augs[strlen(applied_augs) - 1] == '_')
        {
            applied_augs[strlen(applied_augs) - 1] = '\0';
        }

        snprintf(filename, sizeof(filename), "../data/augmented/output_%02d_%s.png", i, applied_augs);

        // Recalculate the exact byte size we need to copy back (it might be 220x220, or 500x500!)
        size_t final_size = current_w * current_h * channels * sizeof(unsigned char);
        cudaMemcpy(h_output, d_src, final_size, cudaMemcpyDeviceToHost);

        // Write the PNG using the dynamic dimensions
        stbi_write_png(filename, current_w, current_h, channels, h_output, current_w * channels);

        // Log it to our dataset map
        fprintf(manifest, "%s, %s\n", filename, applied_augs);
        printf("Generated: %s (Dimensions: %dx%d)\n", filename, current_w, current_h);
    }

    // --- 4. CLEANUP ---
    fclose(manifest);
    cudaFree(d_buffer_A);
    cudaFree(d_buffer_B);
    stbi_image_free(h_input);
    free(h_output);

    printf("\nPipeline Complete! 32 images generated.\n");
    return 0;
}
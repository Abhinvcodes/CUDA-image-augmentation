// include/kernels.h
#ifndef KERNELS_H
#define KERNELS_H

void runCopyKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels);

// Add the new Blur function prototype
void runBlurKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels);

#endif
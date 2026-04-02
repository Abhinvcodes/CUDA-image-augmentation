// include/kernels.h
#ifndef KERNELS_H
#define KERNELS_H

void runCopyKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels);
void runBlurKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels);
void runSharpenKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels);
void runRotateKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels);
void runScaleKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels);

#endif
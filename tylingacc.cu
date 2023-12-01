#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void image_convolution_kernel(const unsigned char *input, unsigned char *output, int width, int height, int channels, const float *kernel, int kernel_size)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int tile_x = bx * TILE_SIZE + tx;
    int tile_y = by * TILE_SIZE + ty;
    int c = bz;

    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2];

    // Load data into the shared memory tile
    #pragma acc loop gang worker vector(32)
    for (int ky = -1; ky <= TILE_SIZE; ky += TILE_SIZE)
    {
        #pragma acc loop gang worker vector(32)
        for (int kx = -1; kx <= TILE_SIZE; kx += TILE_SIZE)
        {
            int ix = tile_x + kx;
            int iy = tile_y + ky;

            // Apply boundary conditions
            ix = max(0, min(width - 1, ix));
            iy = max(0, min(height - 1, iy));

            tile[ty + ky + 1][tx + kx + 1] = input[(iy * width + ix) * channels + c] / 255.0f;
        }
    }

    #pragma acc barrier

    int pad = kernel_size / 2;
    float sum = 0;

    #pragma acc loop gang worker vector(32)
    for (int ky = 0; ky < kernel_size; ++ky)
    {
        #pragma acc loop gang worker vector(32)
        for (int kx = 0; kx < kernel_size; ++kx)
        {
            int ix = tx + kx;
            int iy = ty + ky;

            float pixel = tile[iy][ix];
            float kernel_val = kernel[ky * kernel_size + kx];
            sum += pixel * kernel_val;
        }
    }

    // Write the result back to the output
    output[(tile_y * width + tile_x) * channels + c] = (unsigned char)(sum * 255);
}

int main(int argc, char *argv[])
{
    const char *input_file = "input.jpg";
    const char *output_file = "output.jpg";
    int width, height, channels;

    unsigned char *image_data = stbi_load(input_file, &width, &height, &channels, 0);
    if (!image_data)
    {
        fprintf(stderr, "Error loading image\n");
        return 1;
    }

    float kernel[] = {
        -2, -1, 0,
        -1, 1, 1,
        0, 1, 2};
    int kernel_size = 3;

    unsigned char *d_input, *d_output;
    float *d_kernel;

    size_t image_size = width * height * channels * sizeof(unsigned char);

    cudaMalloc((void **)&d_input, image_size);
    cudaMalloc((void **)&d_output, image_size);
    cudaMalloc((void **)&d_kernel, kernel_size * kernel_size * sizeof(float));

    cudaMemcpy(d_input, image_data, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((width + TILE_SIZE - 1) / TILE_SIZE,
                   (height + TILE_SIZE - 1) / TILE_SIZE,
                   channels);

    #pragma acc data copyin(d_input[0:image_size], d_kernel[0:kernel_size * kernel_size]) copyout(d_output[0:image_size])
    {
        #pragma acc host_data use_device(d_input, d_output, d_kernel)
        {
            image_convolution_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, channels, d_kernel, kernel_size);
        }
    }

    cudaMemcpy(image_data, d_output, image_size, cudaMemcpyDeviceToHost);

    stbi_write_jpg(output_file, width, height, channels, image_data, 100);

    stbi_image_free(image_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}

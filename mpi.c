#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void image_convolution(const unsigned char *input, unsigned char *output, int width, int height, int channels, const float *kernel, int kernel_size)
{
    int pad = kernel_size / 2;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                float sum = 0;
                for (int ky = -pad; ky <= pad; ++ky)
                {
                    for (int kx = -pad; kx <= pad; ++kx)
                    {
                        int ix = x + kx;
                        int iy = y + ky;
                        if (ix >= 0 && ix < width && iy >= 0 && iy < height)
                        {
                            float pixel = input[(iy * width + ix) * channels + c] / 255.0f;
                            float kernel_val = kernel[(ky + pad) * kernel_size + (kx + pad)];
                            sum += pixel * kernel_val;
                        }
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)(sum * 255);
            }
        }
    }
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int width, height, channels;
    unsigned char *image_data = NULL;

    if (world_rank == 0)
    {
        // Load the input image in the root process
        image_data = stbi_load("input.jpg", &width, &height, &channels, 0);
        if (!image_data)
        {
            fprintf(stderr, "Error loading image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image dimensions and channels to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for image data in non-root processes
    if (world_rank != 0)
    {
        image_data = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
    }

    // Broadcast image data to all processes
    MPI_Bcast(image_data, width * height * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int cols_per_process = width / world_size;
    int remaining_cols = width % world_size;

    int start_col = world_rank * cols_per_process;
    int end_col = start_col + cols_per_process;

    if (world_rank == world_size - 1)
    {
        end_col += remaining_cols;
    }

    float kernel[9] = {
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f};


    int kernel_size = 3;

    // Allocate memory for output data
    unsigned char *output_data = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));

    // Perform the convolution for the assigned columns
    int pad = kernel_size / 2;

    double start_time = MPI_Wtime();

    image_convolution(image_data + start_col * channels, output_data + start_col * channels, end_col - start_col, height, channels, kernel, kernel_size);

    // Gather the results from all processes
    unsigned char *gather_output_data = NULL;
    if (world_rank == 0)
    {
        gather_output_data = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
    }

    MPI_Gather(output_data + start_col * channels, (end_col - start_col) * height * channels, MPI_UNSIGNED_CHAR, gather_output_data, (end_col - start_col) * height * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        double end_time = MPI_Wtime();

        // Save the output image
        if (!stbi_write_jpg("outputPad.jpg", width, height, channels, gather_output_data, 100))
        {
            fprintf(stderr, "Error writing output image\n");
            free(image_data);
            free(output_data);
            free(gather_output_data);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Cleanup
        free(image_data);
        free(output_data);
        free(gather_output_data);

        printf("Total time taken: %f seconds\n", end_time - start_time);
    }
    else
    {
        free(image_data);
        free(output_data);
    }

    MPI_Finalize();
    return 0;
}

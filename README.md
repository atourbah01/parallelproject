# parallelproject
Kernel Filtering & Image Covolution
# Parallelizing Convolutions

This project aims to apply the Emboss kernel on images in both serial and parallel. Parallel versions use MPI, OpenMP and CUDA C baisc version and the tiling version.
# File size resolution used 

# input3.jpeg / out3.jpeg (HD resolution)
# input4.jpeg / out4.jpeg (2K resolution)
# input.jpg / out.jpg (4K resolution)
# input2.jpeg / out2.jpeg (5K resolution)
# input1.jpeg / out1.jpeg (8K resolution)

**This code is for an assignment and is not production grade.**

File with results (**time, speedup and efficiency**):

Excel file submitted

## What to expect

The result should be as below:

**Original image:**

input.jpg

**Processed image:**

When running the parallel version, each process will generate part of the final result:

## How it works

the master process will divide the work among slave process each of them taking care of a matrix node of the image pixelated

## How To

doing matrix multiplication between grey scaling and each part of the image for each procces performing a full row of matrix multiplication which makes the image more dense an darker after performing matrix multiplication

### Compile

Compilation shown in word document and power point

### Generate random images

To generate random images with noise for testing purposes, run `./make-images.sh`


### Run experiments
To run experiments and print average time spent processing images generated in the step above, include scripts in every .c or .cu code:

# stb_image.h
#include "stb_image.h"

# stb_image_write.h
#include "stb_image_write.h"

# cuda_runtime.h (only for cuda codes)

## Known bugs
### Distorced images with MPI

*Problem:*  When using the MPI version and both CUDA version files (basic and tiling), the final image looks similar to the input photo.

*Root cause:* Forgot to add conversion from RGB to grayscale.

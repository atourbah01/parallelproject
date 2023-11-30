# parallelproject
Kernel Filtering & Image Covolution
# Parallelizing Convolutions

This project aims to apply the Emboss kernel on images in both serial and parallel. Parallel versions use MPI, OpenMP and CUDA C baisc version and the tiling version.

**This code is for an assignment and is not production grade.**

Spreasheet with results (**time, speedup and efficiency**):

https://docs.google.com/spreadsheets/d/14RXUATwGfS6NzDjerQxIyk-99d9u-4Vin8MdKZ8S0VI/edit?usp=sharing

## What to expect

The result should be as below:

**Original image:**

input.jpg

**Processed image:**

![lena](resources/lena_embossed.bmp)

When running the parallel version, each process will generate part of the final result, as shown in the 4 processes (image files):


**Process 0:**
![lena](Downloads/rank0.bmp)

**Process 1:**
![lena](resources/rank_1.bmp)

**Process 2:**
![lena](resources/rank_2.bmp)

**Process 3:**
![lena](resources/rank_3.bmp)

## How it works
the master process will divide the work among slave process each of them taking care or a row
## How To
doing matrix multiplication between grey scaling and each part of the image for each procces performing a full row of matrix multiplication which makes the image more dense the more matrix multiplication it performs
### Compile
To compile, run `make all`

### Run in serial
To run the serial version on an example image, run:

`./lena_serial.bmp`

### Run in parallel with MPI

To run a parallel version of the algorithm using MPI with 4 processes, run:

`mpirun -n 4 ./lena_mpi.bmp`

To make each process output its partial result, use `verbose` as shown below:

`mpirun -n 4 ./lena_mpi.bmp verbose`

### Run in parallel with MPI and OpenMP
To run a parallel version of the algorithm using MPI an OpenMP with 4 processes, run:

`mpirun -n 4 ./lena_mpi_omp.bmp`



### Generate random images
To generate random images with noise for testing purposes, run `./make-images.sh`


### Run experiments
To run experiments and print average time spent processing images generated in the step above, run `fire` scripts:

``` .sh
# Serial
./fire.sh

# MPI
./fire-mpi.sh

# OpenMP
./fire-omp.sh

# MPI+OpenMP
./fire-mpiomp.sh

```



## Known bugs
### Distorced images with MPI

*Problem:*  When using the MPI version, the final image looks distorted.

*Root cause:* Processes 0 and N-2 deliver more pixels then they should ahte they are used to assemble the final image.

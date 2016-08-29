/**
 *  \file mandelbrot_joe.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

#include <iostream>
#include <cstdlib>

#include "render.hh"
#include <mpi.h>

using namespace std;

#define WIDTH 1000
#define HEIGHT 1000

int mandelbrot(double x, double y) {
	int maxit = 511;
	double cx = x;
	double cy = y;
	double newx, newy;

	int it = 0;
	for (it = 0; it < maxit && (x * x + y * y) < 4; ++it) {
		newx = x * x - y * y + cx;
		newy = 2 * x * y + cy;
		x = newx;
		y = newy;
	}
	return it;
}

int main(int argc, char* argv[]) {

	double minX = -2.1;
	double maxX = 0.7;
	double minY = -1.25;
	double maxY = 1.25;

	int height, width;
	if (argc == 3) {
		height = atoi(argv[1]);
		width = atoi(argv[2]);
		assert(height > 0 && width > 0);
	} else {
		fprintf(stderr, "usage: %s <height> <width>\n", argv[0]);
		fprintf(stderr,
				"where <height> and <width> are the dimensions of the image.\n");
		return -1;
	}

	double it = (maxY - minY) / height;
	double jt = (maxX - minX) / width;
	double x, y;

	int rank, size, offset, blockSize;
	double *recv;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
double time = MPI_Wtime();
	if(rank == 0) {
		recv = (double *) malloc(height * width * sizeof(double));
	}

	blockSize = height / size;
	double *send = (double *) malloc(blockSize * width * sizeof(double));

	y = minY + rank * blockSize * it;
	offset = 0;
	for (int i = rank * blockSize; i < (rank + 1) * blockSize; i++) {
		x = minX;
		for (int j = 0; j < width; j++) {
			send[offset + j] = mandelbrot(x, y) / 512.0;
			x += jt;
		}
		offset += width;
		y += it;
	}
	offset = rank*blockSize;
	MPI_Barrier (MPI_COMM_WORLD);
	MPI_Gather(send, blockSize * width, MPI_DOUBLE, recv + offset, blockSize*width, MPI_DOUBLE, 0, MPI_COMM_WORLD);

double time1 = MPI_Wtime();
if(rank==0) {
cout <<"This code is for joe"<<endl;
cout<<time1-time <<endl;
}
	if (rank == 0) {
		gil::rgb8_image_t img(height, width);
		auto img_view = gil::view(img);

		for (int i = 0; i < size * blockSize; ++i) {
			offset = i * width;
			for (int j = 0; j < width; ++j) {
				img_view(j, i) = render(recv[j + offset]);
			}
		}

		//generate the image for left over rows
		y = minY + size * blockSize * it;
		for (int i = size * blockSize; i < height; i++) {
			x = minX;
			for (int j = 0; j < width; j++) {
				img_view(j, i) = render(mandelbrot(x, y) / 512.0);
				x += jt;
			}
			y += it;

		}

		gil::png_write_view("mandelbrot_joe.png", const_view(img));
	}

	MPI_Finalize();
//double time2 = MPI_Wtime();

}

/* eof */

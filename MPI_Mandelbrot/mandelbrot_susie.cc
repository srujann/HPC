/**
 *  \file mandelbrot_susie.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */
#include <iostream>
#include <cstdlib>
#include "render.hh"
#include <mpi.h>
#include <math.h>


using namespace std;

#define WIDTH 1000
#define HEIGHT 1000

int
mandelbrot(double x, double y) {
  int maxit = 511;
  double cx = x;
  double cy = y;
  double newx, newy;

  int it = 0;
  for (it = 0; it < maxit && (x*x + y*y) < 4; ++it) {
    newx = x*x - y*y + cx;
    newy = 2*x*y + cy;
    x = newx;
    y = newy;
  }
  return it;
}

int
main(int argc, char* argv[]) {
//double time1 = MPI_Wtime();

  double minX = -2.1;
  double maxX = 0.7;
  double minY = -1.25;
  double maxY = 1.25;
  int height, width;
  if (argc == 3) {
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    assert(height > 0 && width > 0);
  }
  else {
    fprintf(stderr, "usage: %s <height> <width>\n", argv[0]);
    fprintf(stderr, "where <height> and <width> are the dimensions of the image.\n");
    return -1;
  }
  double it = (maxY - minY) / height;
  double jt = (maxX - minX) / width;
  double x, y; 
 
  int rank,size,offset;
  double *recv;

  MPI_Init (&argc, &argv);	
  MPI_Comm_size (MPI_COMM_WORLD, &size);	
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
double time1 = MPI_Wtime();
  if (rank == 0) {
    recv = (double*)malloc(sizeof(double) * height * width );
  }

  double *send = (double*) malloc(width * sizeof(double)) ;  

  y = minY+(rank*it);
  for (int i = rank; i < (floor)(height/size) * size; i+=size) {
    x = minX;
    for (int j = 0; j < width; ++j) {
      send[j] = mandelbrot(x, y) / 512.0;
      x += jt;
    }
    y += size*it;
    offset = i*width; //next block of processes 
    MPI_Barrier(MPI_COMM_WORLD); 
    MPI_Gather(send, width, MPI_DOUBLE, recv+offset, width, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
 double time2 = MPI_Wtime();
if(rank==0) {
cout << "This code is for susie" <<endl; 
cout << time2-time1 <<endl;
}

  if (rank==0) {
    gil::rgb8_image_t img(height, width);
    auto img_view = gil::view(img);
    int extraRowSpace = (floor)(height/size) * size; 
    for (int i = 0; i < extraRowSpace; ++i) {
      offset = i*width;
      for (int j = 0; j < width; ++j) {
        img_view(j, i) = render(recv[j+offset]);
      }
    }

   //generate the image for left over rows
   y = minY + (extraRowSpace * it);
   for( int  i =extraRowSpace; i<height; i++)
   {
      x = minX;
     for (int j = 0; j < width; j++) {
	img_view(j, i) = render(mandelbrot(x, y) / 512.0);
	x += jt;
      }
      y += it;
    }


    gil::png_write_view("mandelbrot_susie.png", const_view(img));
  }
  MPI_Finalize();
 // double time2 = MPI_Wtime();


  return 0;
}

/* eof */
#include "solver.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void create_problem(int nx, int ny, double alpha, int NX, int NY, heat_problem * pb) {
	pb->nx = nx/NX + 2;
	pb->ny = ny/NY + 2;
	pb->dx = 1.0/(nx+1);
	pb->dy = 1.0/(ny+1);
	pb->alpha = alpha;
	pb->T = calloc(pb->nx*pb->ny, sizeof(double));
	for (int i = 0; i < pb->ny; i++) {
		pb->T[i*pb->nx] = 1.0; 
		pb->T[i*pb->nx+(pb->nx-1)] = 1.0; 
	}
	for (int j = 0; j < pb->nx; j++) {
		pb->T[j] = 0.0; 
		pb->T[(pb->ny-1)*pb->nx+j] = 0.0; 
	}
	//pb->ycomm = ...;
	//pb->ycomm_rank, pb->ycomm_size à renseigner
}

void free_problem(heat_problem * pb) {
	free(pb->T);
}

void step(heat_problem * pb, double dt) {
	int nx = pb->nx;
	int ny = pb->ny;
	int size = nx * ny * sizeof(double);
	double * oldT = malloc(size);
	memcpy(oldT, pb->T, size);

	for (int i = 1; i < ny-1; i++) {
		for (int j = 1; j < nx-1; j++) {
			int index = i*nx + j;
			double lapl_x = (oldT[index+1] - 2*oldT[index] + oldT[index-1])/(pb->dx*pb->dx);
			double lapl_y = (oldT[index+nx] - 2*oldT[index] + oldT[index-nx])/(pb->dy*pb->dy);
			pb->T[index] += pb->alpha * dt * (lapl_x + lapl_y);
		}
	}
	free(oldT);
}

void print_result(heat_problem * pb) {
	int nx = pb->nx;
	int ny = pb->ny;
	for (int i = 0; i < ny; i++) {
		for (int j = 0; j < nx; j++) {
			printf("%3.2f ", pb->T[i*nx + j]);
		}
		printf("\n");
	}
}

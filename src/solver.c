#include "solver.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


/** Retourne la différence (en secondes) entre deux timespec */
double get_delta(struct timespec begin, struct timespec end) {
	return end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) * 1e-9;
}


void create_problem(int nx, int ny, double alpha, int NX, int NY, heat_problem * pb) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
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
	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("Temps passé dans create_problem: %f seconds\n", get_delta(start, end));
}

void free_problem(heat_problem * pb) {
	free(pb->T);
}


// void step(heat_problem * pb, double dt) {
// 	int nx = pb->nx;
// 	int ny = pb->ny;
// 	int size = nx * ny * sizeof(double);
// 	double * oldT = malloc(size);
// 	memcpy(oldT, pb->T, size);

// 	for (int i = 1; i < ny-1; i++) {
// 		for (int j = 1; j < nx-1; j++) {
// 			int index = i*nx + j;
// 			double lapl_x = (oldT[index+1] - 2*oldT[index] + oldT[index-1])/(pb->dx*pb->dx);
// 			double lapl_y = (oldT[index+nx] - 2*oldT[index] + oldT[index-nx])/(pb->dy*pb->dy);
// 			pb->T[index] += pb->alpha * dt * (lapl_x + lapl_y);
// 		}
// 	}
// 	free(oldT);
// }


void step(heat_problem *pb, double dt) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = pb->nx;
    int ny = pb->ny;
    int ny_local = ny / size;

    // Allocation locale
    double *oldT = malloc(nx * ny_local * sizeof(double));
    memcpy(oldT, pb->T_local, nx * ny_local * sizeof(double));

    // Halo exchange
    if (rank > 0) {
        MPI_Sendrecv(oldT, nx, MPI_DOUBLE, rank-1, 0,
                     oldT - nx, nx, MPI_DOUBLE, rank-1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size-1) {
        MPI_Sendrecv(oldT + (ny_local-1)*nx, nx, MPI_DOUBLE, rank+1, 0,
                     oldT + ny_local*nx, nx, MPI_DOUBLE, rank+1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Calcul local
    for (int i = 1; i < ny_local-1; i++) {
        for (int j = 1; j < nx-1; j++) {
            int index = i*nx + j;
            double lapl_x = (oldT[index+1] - 2*oldT[index] + oldT[index-1])/(pb->dx*pb->dx);
            double lapl_y = (oldT[index+nx] - 2*oldT[index] + oldT[index-nx])/(pb->dy*pb->dy);
            pb->T_local[index] += pb->alpha * dt * (lapl_x + lapl_y);
        }
    }

    free(oldT);

    // Rapatriement des résultats
    MPI_Gather(pb->T_local, nx*ny_local, MPI_DOUBLE,
               pb->T, nx*ny_local, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
}


void print_result(heat_problem * pb) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	int nx = pb->nx;
	int ny = pb->ny;
	for (int i = 0; i < ny; i++) {
		for (int j = 0; j < nx; j++) {
			printf("%3.2f ", pb->T[i*nx + j]);
		}
		printf("\n");
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("Temps passé dans print_result: %f seconds\n", get_delta(start, end));
}

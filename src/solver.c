#include "solver.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


/** Retourne la différence (en secondes) entre deux timespec */
double get_delta(struct timespec begin, struct timespec end) {
	return end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) * 1e-9;
}


void create_problem(int nx, int ny, double alpha, int NX, int NY, heat_problem *pb) {
    // struct timespec start, end;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    // Dimensions locales (avec fantômes)
    pb->nx = nx / NX + 2;   // +2 pour colonnes fantômes gauche/droite
    pb->ny = ny / NY + 2;   // +2 pour lignes fantômes haut/bas
    pb->dx = 1.0 / (nx + 1);
    pb->dy = 1.0 / (ny + 1);
    pb->alpha = alpha;

    // Allocation du tableau local
    pb->T = calloc(pb->nx * pb->ny, sizeof(double));

    // Conditions aux limites verticales (x=0 et x=nx-1)
    for (int i = 0; i < pb->ny; i++) {
        pb->T[i * pb->nx] = 1.0;                  // bord gauche
        pb->T[i * pb->nx + (pb->nx - 1)] = 1.0;   // bord droit
    }

    // Conditions aux limites horizontales (y=0 et y=ny-1)
    for (int j = 0; j < pb->nx; j++) {
        pb->T[j] = 0.0;                           // bord haut
        pb->T[(pb->ny - 1) * pb->nx + j] = 0.0;   // bord bas
    }

    // Création du communicateur parallèle en Y
    pb->ycomm = MPI_COMM_WORLD;
    MPI_Comm_rank(pb->ycomm, &pb->ycomm_rank);
    MPI_Comm_size(pb->ycomm, &pb->ycomm_size);

    // clock_gettime(CLOCK_MONOTONIC, &end);
    // printf("Temps passé dans create_problem: %f seconds\n", get_delta(start, end));
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


void step_parallel(heat_problem *pb, double dt) {
    int rank = pb->ycomm_rank;
    int size = pb->ycomm_size;
    int nx = pb->nx;
    int ny = pb->ny;

    int north = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int south = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // 1) Échange des fantômes sur T_n
    MPI_Sendrecv(&pb->T[1 * nx], nx, MPI_DOUBLE, north, 0,
                 &pb->T[(ny - 1) * nx], nx, MPI_DOUBLE, south, 0,
                 pb->ycomm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&pb->T[(ny - 2) * nx], nx, MPI_DOUBLE, south, 1,
                 &pb->T[0], nx, MPI_DOUBLE, north, 1,
                 pb->ycomm, MPI_STATUS_IGNORE);

    // 2) Copie T_n
    double *oldT = malloc(nx * ny * sizeof(double));
    memcpy(oldT, pb->T, nx * ny * sizeof(double));

    // 3) Calcul local
    for (int i = 1; i < ny - 1; i++) {
        for (int j = 1; j < nx - 1; j++) {
            int id = i * nx + j;
            double lapl_x = (oldT[id+1] - 2*oldT[id] + oldT[id-1])/(pb->dx*pb->dx);
            double lapl_y = (oldT[id+nx] - 2*oldT[id] + oldT[id-nx])/(pb->dy*pb->dy);
            pb->T[id] = oldT[id] + pb->alpha * dt * (lapl_x + lapl_y);
        }
    }

    free(oldT);
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
	// printf("Temps passé dans print_result: %f seconds\n", get_delta(start, end));
}

void print_mean(heat_problem *pb, int ny_global) {
    int nx = pb->nx;
    int ny_local = pb->ny - 2;

    double *local_sum = calloc(nx, sizeof(double));
    double *global_sum = NULL;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // On s'assure que seul le rang 0 alloue le tableau global
    if (world_rank == 0) {
        global_sum = calloc(nx, sizeof(double));
    }

    // Somme locale (hors fantômes)
    for (int i = 1; i <= ny_local; i++) {
        for (int j = 0; j < nx; j++) {
            local_sum[j] += pb->T[i * nx + j];
        }
    }

    MPI_Reduce(
        local_sum,
        global_sum,
        nx,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    // Affichage de la moyenne par le rang 0
    if (world_rank == 0) {
        for (int j = 0; j < nx; j++) {
            printf("%6.4f ", global_sum[j] / ny_global);
        }
        printf("\n");
        free(global_sum);
    }

    free(local_sum);
}


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

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	pb->ycomm_rank = rank / NX;
	pb->ycomm_size = NY;
	pb->xcomm_rank = rank % NX;
	pb->xcomm_size = NX;

    // // Conditions aux limites verticales (x=0 et x=nx-1)
    // for (int i = 0; i < pb->ny; i++) {
    //     pb->T[i * pb->nx] = 1.0;                  // bord gauche
    //     pb->T[i * pb->nx + (pb->nx - 1)] = 1.0;   // bord droit
    // }

    // // Conditions aux limites horizontales (y=0 et y=ny-1)
    // for (int j = 0; j < pb->nx; j++) {
    //     pb->T[j] = 0.0;                           // bord haut
    //     pb->T[(pb->ny - 1) * pb->nx + j] = 0.0;   // bord bas
    // }

	// Conditions aux limites sur le bord gauche
	if (pb->xcomm_rank == 0) {
		for (int i = 0; i < pb->ny; i++)
			pb->T[i * pb->nx] = 1.0;
	}

	// Conditions aux limites sur le bord droit
	if (pb->xcomm_rank == pb->xcomm_size - 1) {
		for (int i = 0; i < pb->ny; i++)
			pb->T[i * pb->nx + pb->nx - 1] = 1.0;
	}

	// Conditions aux limites sur le bord haut
	if (pb->ycomm_rank == 0) {
		for (int j = 0; j < pb->nx; j++)
			pb->T[j] = 0.0;
	}

	// Conditions aux limites sur le bord bas
	if (pb->ycomm_rank == pb->ycomm_size - 1) {
		for (int j = 0; j < pb->nx; j++)
			pb->T[(pb->ny - 1) * pb->nx + j] = 0.0;
	}

	// Communicateur par ligne Y
	MPI_Comm_split(MPI_COMM_WORLD, pb->ycomm_rank, pb->xcomm_rank, &pb->ycomm);
	MPI_Comm_rank(pb->ycomm, &pb->ycomm_rank);
	MPI_Comm_size(pb->ycomm, &pb->ycomm_size);

	// Communicateur par colonne X
	MPI_Comm_split(MPI_COMM_WORLD, pb->xcomm_rank, pb->ycomm_rank, &pb->xcomm);
	MPI_Comm_rank(pb->xcomm, &pb->xcomm_rank);
	MPI_Comm_size(pb->xcomm, &pb->xcomm_size);

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
    int nx = pb->nx;
    int ny = pb->ny;
    int yrank = pb->ycomm_rank;
    int size = pb->ycomm_size;

    int north = (yrank == 0) ? MPI_PROC_NULL : yrank - 1;
    int south = (yrank == size - 1) ? MPI_PROC_NULL : yrank + 1;

    // 1) Échange des fantômes sur T_n
    MPI_Sendrecv(&pb->T[1 * nx], nx, MPI_DOUBLE, north, 0,
                 &pb->T[(ny - 1) * nx], nx, MPI_DOUBLE, south, 0,
                 pb->ycomm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&pb->T[(ny - 2) * nx], nx, MPI_DOUBLE, south, 1,
                 &pb->T[0], nx, MPI_DOUBLE, north, 1,
                 pb->ycomm, MPI_STATUS_IGNORE);

	// Échange selon X
	int xrank = pb->xcomm_rank;
	int xsize = pb->xcomm_size;
	int west  = (xrank == 0) ? MPI_PROC_NULL : xrank - 1;
	int east  = (xrank == xsize - 1) ? MPI_PROC_NULL : xrank + 1;

	// Nécessité de créer un type MPI pour envoyer/recevoir des colonnes
	MPI_Datatype column_type;
	MPI_Type_vector(
		ny - 2,
		1, 
		nx,
		MPI_DOUBLE,
		&column_type
	);
	MPI_Type_commit(&column_type);

	// Envoi vers l'ouest, réception depuis l'est
	MPI_Sendrecv(
		&pb->T[1 * nx + 1],           // colonne intérieure gauche
		1, column_type,
		west, 2,
		&pb->T[1 * nx + (nx - 1)],    // colonne fantôme droite
		1, column_type,
		east, 2,
		pb->xcomm, MPI_STATUS_IGNORE
	);

	// Envoi vers l'est, réception depuis l'ouest
	MPI_Sendrecv(
		&pb->T[1 * nx + (nx - 2)],    // colonne intérieure droite
		1, column_type,
		east, 3,
		&pb->T[1 * nx + 0],           // colonne fantôme gauche
		1, column_type,
		west, 3,
		pb->xcomm, MPI_STATUS_IGNORE
	);


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


// void print_result(heat_problem * pb) {
// 	struct timespec start, end;
// 	clock_gettime(CLOCK_MONOTONIC, &start);
// 	int nx = pb->nx;
// 	int ny = pb->ny;
// 	for (int i = 0; i < ny; i++) {
// 		for (int j = 0; j < nx; j++) {
// 			printf("%3.2f ", pb->T[i*nx + j]);
// 		}
// 		printf("\n");
// 	}
// 	clock_gettime(CLOCK_MONOTONIC, &end);
// 	// printf("Temps passé dans print_result: %f seconds\n", get_delta(start, end));
// }

void print_result(heat_problem * pb, int rank) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
    int nx = pb->nx;
    int ny = pb->ny;
	
    printf("Processus %d:\n", rank);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%6.2f ", pb->T[i*nx + j]);
        }
        printf("\n");
    }
    printf("\n");
	clock_gettime(CLOCK_MONOTONIC, &end);
}


void print_mean(heat_problem *pb) {
    int nx = pb->nx;
    int ny = pb->ny;

    double *local_sum = calloc(nx, sizeof(double));
    double *global_sum = NULL;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        global_sum = calloc(nx, sizeof(double));
    }

    // Somme locale sur toutes les lignes, y compris fantômes
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            local_sum[j] += pb->T[i * nx + j];
        }
    }

    MPI_Reduce(local_sum, global_sum, nx, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        for (int j = 0; j < nx; j++) {
            printf("%6.2f ", global_sum[j] / (ny * 4)); // diviser par nb total de lignes (nb processus = 4 ici)
        }
        printf("\n");
        free(global_sum);
    }

    free(local_sum);
}



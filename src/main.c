#include "solver.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char * argv[]) {
	MPI_Init(&argc, &argv);
	int comm_rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	if (argc < 3) {
		if (comm_rank == 0) {
			printf("USAGE: %s <nx> <ny>\n", argv[0]);
		}
		MPI_Finalize();
		return -1;
	}
	printf("Processus %d sur %d\n", comm_rank, comm_size);


	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);

	int NX = 1; // Nombre de domaines selon X (Q3.2)
	int NY = comm_size;

	heat_problem pb;
	create_problem(nx, ny, 1.0, NX, NY, &pb);

	double dt = 0.1*pb.dx*pb.dy;
	int niter = 1000;

	// Temps total passé dans le calcul
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

	for (int i = 0; i < niter; i++) {
		step_parallel(&pb, dt); // Appel à la version parallèle
		// On appelle print_mean tous les 100 itérations
		if (i % 100 == 0){
			print_mean(&pb, ny);
		}
	}

	// ------------------ Rapatriement du champ global ------------------

	int local_inner_ny = pb.ny - 2;
	int local_inner_size = local_inner_ny * pb.nx;

	int nx_global = pb.nx;
	int ny_global = local_inner_ny * comm_size + 2;

	double *T_global = NULL;
	if (comm_rank == 0) {
		T_global = calloc(nx_global * ny_global, sizeof(double));
	}

	MPI_Gather(&pb.T[pb.nx], local_inner_size, MPI_DOUBLE, T_global + nx_global, local_inner_size,
		MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (comm_rank == 0) {
		pb.T = T_global;
		pb.ny = ny_global;

		// print_result(&pb);
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("Temps total passé dans toutes les itérations de calcul: %f seconds\n", get_delta(start, end));
	// print_result(&pb);

	free_problem(&pb);
	MPI_Finalize();

	
	return 0;
}
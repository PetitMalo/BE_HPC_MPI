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
	if (argc < 4) {
		if (comm_rank == 0) {
			printf("USAGE: %s <nx> <ny> <NX>\n", argv[0]);
		}
		MPI_Finalize();
		return -1;
	}
	printf("Processus %d sur %d\n", comm_rank, comm_size);


	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);

	int NX = atoi(argv[3]); // Nombre de domaines selon X (Q3.1)
	if (NX <= 0 || comm_size % NX != 0) {
		if (comm_rank == 0) {
			printf("Erreur : NX doit diviser le nombre de processus\n");
		}
		MPI_Finalize();
		return -1;
	}
	int NY = comm_size / NX; // Nombre de domaines selon Y

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
			print_mean(&pb);
		}
	}

	// ------------------ Rapatriement du champ global ------------------
	int nx_global = pb.nx;
	int ny_global = pb.ny * comm_size;  // tout inclure, fantômes compris

	double *T_global = NULL;
	if (comm_rank == 0) {
		T_global = calloc(nx_global * ny_global, sizeof(double));
	}

	// On récupère tout le tableau local
	MPI_Gather(pb.T, pb.nx * pb.ny, MPI_DOUBLE, T_global, pb.nx * pb.ny,
			MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (comm_rank == 0) {
		pb.T = T_global;
		pb.ny = ny_global;

		print_result(&pb, 0);
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("Temps total passé dans toutes les itérations de calcul: %f seconds\n", get_delta(start, end));
	printf("Total time for %d iterations: %f seconds\n", niter, get_delta(start, end));

	free_problem(&pb);
	MPI_Finalize();

	
	return 0;
}

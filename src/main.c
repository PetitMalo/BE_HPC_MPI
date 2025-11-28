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


	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);

	int NX = 1; // Nombre de domaines selon X (Q3.2)
	int NY = comm_size;

	heat_problem pb;
	create_problem(nx, ny, 1.0, NX, NY, &pb);

	double dt = 0.1*pb.dx*pb.dy;
	int niter = 1000;
	// Temps total passé dans les itérations de calcul
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (int i = 0; i < niter; i++) {
		step(&pb, dt);
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("Temps total passé dans toutes les itérations de calcul: %f seconds\n", get_delta(start, end));

	// print_result(&pb);

	free_problem(&pb);
	MPI_Finalize();
	return 0;
}
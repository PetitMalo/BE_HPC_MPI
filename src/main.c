#include "solver.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char * argv[]) {
	MPI_Init(&argc, &argv);
	int comm_rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_siz	e);
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
	for (int i = 0; i < niter; i++) {
		step(&pb, dt);
	}

	print_result(&pb);

	free_problem(&pb);
	MPI_Finalize();
	return 0;
}

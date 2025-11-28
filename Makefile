CC=mpicc
CFLAGS=-Wall -Werror -std=gnu99 -fopenmp -Iinclude -g -D_POSIX_C_SOURCE=200809L
LDFLAGS=-fopenmp -lrt -lm

solveur : main.o solver.o
	$(CC) -o $@ $^ $(LDFLAGS)

%.o : src/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean :
	rm -f *.o solveur

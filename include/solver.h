#ifndef SOLVER_H
#define SOLVER_H

#include <mpi.h>
#include <time.h>

// Définition du problème local
typedef struct {
	int nx; // Nombre de points dans le maillage local
	int ny; // EN COMPTANT les conditions aux limites
	double dx; // Pas d'espace
	double dy; // entre deux points du maillage
	double alpha; // Conductivité thermique
	double * T; // Tableau 2D local des valeurs de températures (taille nx*ny)
	MPI_Comm ycomm; // Communicateur pour la parallélisation selon y
	int ycomm_rank; // Rang du processus selon y
	int ycomm_size; // Taille du communicateur selon y
	// On pourra ajouter un communicateur selon X en partie 3.
} heat_problem;

/**
 * Initialise le problème local.
 * @param nx,ny Nombre de points du maillage global HORS conditions aux limites
 * @param alpha Conductivité thermique
 * @param NX,NY Nombre de sous-domaines selon chaque dimension
 * @param pb Problème initialisé 
 */
void create_problem(int nx, int ny, double alpha, int NX, int NY, heat_problem * pb);

/**
 * Libère la mémoire initialisée lors de create_problem.
 */
void free_problem(heat_problem * pb);

/**
 * Calcule une itération.
 * @param pb Problème
 * @param dt Pas de temps
 */
void step(heat_problem * pb, double dt);

/**
 * Affiche le résultat du problème.
 */
void print_result(heat_problem * pb);


/** Retourne la différence (en secondes) entre deux timespec */
double get_delta(struct timespec begin, struct timespec end);
#endif

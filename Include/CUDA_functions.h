#pragma once
#include "Defines.cuh"

/*-----------------------------------------------------------*/
/*              Repulsive force evaluation (GPU)             */
/*-----------------------------------------------------------*/
DLL_API void pairwise_gpu(Eigen::ArrayXf *force_x, Eigen::ArrayXf *force_y, Eigen::ArrayXf atoms_x, 
	Eigen::ArrayXf atoms_y, float SD_factor, int threads_per_block);

/*-----------------------------------------------------------*/
/*              Repulsive force evaluation (GPU)             */
/*-----------------------------------------------------------*/
DLL_API void tracker_gpu(Eigen::ArrayXXf *G, Eigen::ArrayXf *p, Eigen::ArrayXf atoms_x, Eigen::ArrayXf atoms_y,
	const int n_pop, const int n_grids, const int threads_per_block);
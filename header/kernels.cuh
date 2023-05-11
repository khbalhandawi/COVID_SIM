#pragma once

#ifndef __KERNELS__
#define __KERNELS__

namespace CUDA_GPU {

	/*-----------------------------------------------------------*/
	/*     Compute pairwise distance matrix matrix (kernel)      */
	/*-----------------------------------------------------------*/
	__global__ void calc_distances(float* distances, float* atoms_x, float* atoms_y, int N);

	/*-----------------------------------------------------------*/
	/*          Compute repulsive force matrix (kernel)          */
	/*-----------------------------------------------------------*/
	__global__ void calc_force_m(float* diffs_x, float* diffs_y, float* atoms_x, float* atoms_y, float SD_factor, int N);

	/*-----------------------------------------------------------*/
	/*           Reduce repulsive force matrix (kernel)          */
	/*-----------------------------------------------------------*/
	__global__ void calc_forces(float* output, float* matrix, int N_rows, int N_cols, int N_sum);

	/*-----------------------------------------------------------*/
	/*              Compute tracking matrix (kernel)             */
	/*-----------------------------------------------------------*/
	__global__ void calc_tracking_matrix(float* G, float* G_track, float* atoms_x, float* atoms_y, int n_grids, int N_rows, int N_cols);

	/*-----------------------------------------------------------*/
	/*      Initialize any device vector with val (kernel)       */
	/*-----------------------------------------------------------*/
	template<typename T>
	__global__ void initKernel(T * devPtr, const T val, const size_t nwords)
	{
		int tidx = threadIdx.x + blockDim.x * blockIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (; tidx < nwords; tidx += stride)
			devPtr[tidx] = val;
	}

}

#endif
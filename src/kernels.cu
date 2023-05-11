/*---------------------------------------------------------------------------------*/
/*  COVID GPU - GPU accelerated functions for agent-based modelling -              */
/*                                                                                 */
/*  COVID GPU - version 1.0.0 has been created by                                  */
/*                 Khalil Al Handawi           - McGill University                 */
/*                                                                                 */
/*  The copyright of COVID_SIM_GPU is owned by                                     */
/*                 Khalil Al Handawi           - McGill University                 */
/*                                                                                 */
/*                                                                                 */
/*  Contact information:                                                           */
/*    McGill University - Systems Optimization Lab (SOL)                           */
/*    Macdonald Engineering Building, 817 Sherbrooke Street West,                  */
/*    Montreal (Quebec) H3A 0C3 Canada                                             */
/*    e-mail: khalil.alhandawi@mail.mcgill.ca                                      */
/*    phone : 1-514-398-2343                                                       */
/*                                                                                 */
/*  This program is free software: you can redistribute it and/or modify it        */
/*  under the terms of the GNU Lesser General Public License as published by       */
/*  the Free Software Foundation, either version 3 of the License, or (at your     */
/*  option) any later version.                                                     */
/*                                                                                 */
/*  This program is distributed in the hope that it will be useful, but WITHOUT    */
/*  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or          */
/*  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License    */
/*  for more details.                                                              */
/*                                                                                 */
/*  You should have received a copy of the GNU Lesser General Public License       */
/*  along with this program. If not, see <http://www.gnu.org/licenses/>.           */
/*                                                                                 */
/*---------------------------------------------------------------------------------*/

/**
 \file   kernels.cu
 \brief  GPU kernels (implementation)
 \author Khalil Al Handawi
 \date   2021-01-11
 \see    kernels.cuh
 */

#include "kernels.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// TODO: convert all kernels to grid-stride loop paradigm
/*-----------------------------------------------------------*/
/*     Compute pairwise distance matrix matrix (kernel)      */
/*-----------------------------------------------------------*/
 __global__ void CUDA_GPU::calc_distances(float* distances, float* atoms_x, float* atoms_y, int N)
 {
 	int i(threadIdx.x + blockIdx.x * blockDim.x);
 	int j(threadIdx.y + blockIdx.y * blockDim.y);
 
 	if (i >= N || j >= N) {
 		return;
 	}
 
 	distances[i + N * j] =
 		(atoms_x[i] - atoms_x[j]) * (atoms_x[i] - atoms_x[j]) +
 		(atoms_y[i] - atoms_y[j]) * (atoms_y[i] - atoms_y[j]);
 }

/*-----------------------------------------------------------*/
/*          Compute repulsive force matrix (kernel)          */
/*-----------------------------------------------------------*/
__global__ void CUDA_GPU::calc_force_m(float* diffs_x, float* diffs_y, float* atoms_x, float* atoms_y, float SD_factor, int N)
{
	int i(threadIdx.x + blockIdx.x * blockDim.x);
	int j(threadIdx.y + blockIdx.y * blockDim.y);

	if (i >= N || j >= N) {
		return;
	}

	float distance = ((atoms_x[i] - atoms_x[j]) * (atoms_x[i] - atoms_x[j]) +
		(atoms_y[i] - atoms_y[j]) * (atoms_y[i] - atoms_y[j]));
	float distance3 = distance * distance * distance + 1e-15;

	diffs_x[i + N * j] = -SD_factor * (atoms_x[i] - atoms_x[j]) / distance3;

	diffs_y[i + N * j] = -SD_factor * (atoms_y[i] - atoms_y[j]) / distance3;
}

/*-----------------------------------------------------------*/
/*           Reduce repulsive force matrix (kernel)          */
/*-----------------------------------------------------------*/
__global__ void CUDA_GPU::calc_forces(float* output, float* matrix, int N_rows, int N_cols, int N_sum)
{

	size_t start_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t start_y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t stride_x = blockDim.x * gridDim.x;
	size_t stride_y = blockDim.y * gridDim.y;
	size_t Mx = blockDim.x;
	size_t My = blockDim.y;
	extern __shared__ float partial[];

	for (int i = start_x; i < N_sum; i += stride_x) {
		for (int j = start_y; j < N_cols; j += stride_y) {

			// each thread loads one element from global to shared mem
			partial[threadIdx.x + threadIdx.y*Mx] = matrix[i + j*N_rows];
			__syncthreads();

			// do reduction in shared mem
			for(unsigned int s=1; s < blockDim.x; s *= 2) {
				if (threadIdx.x % (2*s) == 0) {
					partial[threadIdx.x + threadIdx.y*Mx] += partial[(threadIdx.x+s) + threadIdx.y*Mx];
				}
				__syncthreads();
			}
			// write result for this block to global mem
			if (threadIdx.x == 0) {
				output[blockIdx.x + j*N_rows] = partial[threadIdx.y*Mx];
			}
		}
	}
}

/*-----------------------------------------------------------*/
/*              Compute tracking matrix (kernel)             */
/*-----------------------------------------------------------*/
__global__ void CUDA_GPU::calc_tracking_matrix(float* G, float* G_track, float* atoms_x, float* atoms_y, int n_grids, int N_rows, int N_cols)
{
	int i(threadIdx.x + blockIdx.x * blockDim.x);
	int j(threadIdx.y + blockIdx.y * blockDim.y);

	if (i >= N_rows || j >= N_cols) 
	{
		return;
	}

	float g1 = (float)(j % n_grids) / float(n_grids);
	float g2 = (float)(j / n_grids) / float(n_grids);
	float g3 = (float)((j % n_grids) + 1) / float(n_grids);
	float g4 = (float)((j / n_grids) + 1) / float(n_grids);

	bool check = (atoms_x[i] > g1) && (atoms_y[i] > g2) && (atoms_x[i] <= g3) && (atoms_y[i] <= g4);

	float t = (check) ? 1 : 0;
	G[i + j * N_rows] += t; // update tracing matrix

	float track = (check) ? 1 : G_track[i + j * N_rows];
	G_track[i + j * N_rows] = track; // update tracking matrix

}
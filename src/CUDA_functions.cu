/*---------------------------------------------------------------------------------*/
/*  COVID GPU - GPU accelerated functions for agent-based modelling -              */
/*                                                                                 */
/*  COVID GPU - version 1.0.0 has been created by                                  */
/*                 Khalil Al Handawi           - McGill University                 */
/*                                                                                 */
/*  The copyright of NOMAD - version 3.9.1 is owned by                             */
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
 \file   CUDA_functions.cu
 \brief  GPU accelerate matrix functions (implementation)
 \author Khalil Al Handawi
 \date   2021-01-11
 \see    CUDA_functions.cu
 */
#include  "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <Eigen/Core>

#include "CUDA_functions.cuh"
#include "kernels.cuh"
#include "Utilities.cuh"

using namespace std;

/*-----------------------------------------------------------*/
/*                    CUDA ERROR CHECKING                    */
/*-----------------------------------------------------------*/
inline void _check(cudaError_t code, char *file, int line)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

/*-----------------------------------------------------------*/
/*                CUDA ERROR CHECKING (macro)                */
/*-----------------------------------------------------------*/
#define check(ans) { _check((ans), __FILE__, __LINE__); }

/*-----------------------------------------------------------*/
/*						 Constructor					     */
/*-----------------------------------------------------------*/
CUDA_GPU::Kernels::Kernels(const int n_pop, const int n_grids, const int threads_per_block_in, const cublasHandle_t handle)
{
	N_rows = n_pop; // number of rows
	N_cols = n_grids * n_grids; // number of rows
	N_grids = n_grids;
	threads_per_block = threads_per_block_in;

	bytes = N_rows * sizeof(float); // Size, in bytes, of each vector

	check(cudaMallocHost(&atoms_x_h, bytes));
	check(cudaMallocHost(&atoms_y_h, bytes));
	check(cudaMalloc(&atoms_x_d, bytes));
	check(cudaMalloc(&atoms_y_d, bytes));

	//======================================================//
	// Pairwise distance and difference calculation

	check(cudaMalloc(&diffs_x_d, N_rows * bytes));
	check(cudaMalloc(&diffs_y_d, N_rows * bytes));

	//======================================================//
	// Force outputs

	check(cudaMalloc(&force_x_d, bytes));
	check(cudaMalloc(&force_y_d, bytes));

	check(cudaMallocHost(&force_x_h, bytes));
	check(cudaMallocHost(&force_y_h, bytes));

	//======================================================//
	// Ground covered outputs

	bytes_rows = N_rows * sizeof(float); // Size, in bytes, of each vector
	bytes_cols = N_cols * sizeof(float); // Size, in bytes, of each vector

	check(cudaMallocHost(&G_h, N_rows * N_cols * sizeof(float))); // tracing matrix (host)
	check(cudaMalloc(&G_d, N_rows * N_cols * sizeof(float))); // tracing matrix (device)

	check(cudaMallocHost(&G_track_h, N_rows * N_cols * sizeof(float))); // tracking matrix (host)
	check(cudaMalloc(&G_track_d, N_rows * N_cols * sizeof(float))); // tracking matrix (device)
	//======================================================//
	// Percentage covered outputs

	check(cudaMalloc(&p_d, bytes_rows));
	check(cudaMallocHost(&p_h, bytes_rows));

#ifndef CUBLAS_NDEBUG
	//======================================================//
	// Rowewise multiplication initialization
	const float value = 1.f;
	// initialize vector with ones using CUDA (for force matrix multiplication)
	check(cudaMalloc(&d_ones_force, bytes));

	// initialize vector with ones using CUDA (for tracking matrix multiplication)
	check(cudaMalloc(&d_ones_track, bytes));
	int n_blocks_cols(div_up(N_cols, sqrt(threads_per_block)));
	CUDA_GPU::initKernel << <n_blocks_cols, threads_per_block >> > (d_ones_track, value, N_cols);

	local_handle = handle;
	//cublascheck(cublasCreate(&handle)); // construct cublas handle

	alpha = 1.f;
	beta = 0.f;
#endif

}

/*-----------------------------------------------------------*/
/*              Repulsive force evaluation (GPU)             */
/*-----------------------------------------------------------*/
void CUDA_GPU::Kernels::pairwise_gpu(Eigen::ArrayXf atoms_x, Eigen::ArrayXf atoms_y, float SD_factor)
{

	N_actual = atoms_x.rows(); // actual number of elements

	//=======================================//
	//          Transfer values....          //
	//=======================================//

	Eigen::ArrayXf::Map(atoms_x_h, atoms_x.rows()) = atoms_x; // Map to x vector (host)
	Eigen::ArrayXf::Map(atoms_y_h, atoms_y.rows()) = atoms_y; // Map to y vector (host)

	check(cudaMemcpy(atoms_x_d, atoms_x_h, bytes, cudaMemcpyHostToDevice));
	check(cudaMemcpy(atoms_y_d, atoms_y_h, bytes, cudaMemcpyHostToDevice));

	//======================================================//
	// Matrix grids
	int n_blocks(div_up(N_actual, sqrt(threads_per_block)));

	dim3 blockSize = dim3(sqrt(threads_per_block), sqrt(threads_per_block));
	dim3 gridSize = dim3(n_blocks, n_blocks);

	//======================================================//
	// Pairwise distance and difference calculation

	CUDA_GPU::calc_force_m << <gridSize, blockSize >> > (diffs_x_d, diffs_y_d, atoms_x_d, atoms_y_d, SD_factor, N_actual);
	//======================================================//
	// Force calculation (rowise matrix reduction by CUBLAS)


#ifndef CUBLAS_NDEBUG
	//======================================================//
	// Rowewise multiplication initialization
	const float value = 1.f;

	// initialize vector with ones using CUDA (for force matrix multiplication)
	int n_blocks_rows(div_up(N_actual, sqrt(threads_per_block)));
	CUDA_GPU::initKernel << <n_blocks_rows, threads_per_block >> > (d_ones_force, value, N_actual);

	cublascheck(cublasSgemv(local_handle, CUBLAS_OP_T, N_actual, N_actual, &alpha, diffs_x_d, N_actual,
		d_ones_force, 1, &beta, force_x_d, 1)); // rowwise multiplication x
	cublascheck(cublasSgemv(local_handle, CUBLAS_OP_T, N_actual, N_actual, &alpha, diffs_y_d, N_actual,
		d_ones_force, 1, &beta, force_y_d, 1)); // rowwise multiplication y
#endif
	//======================================================//

	check(cudaPeekAtLastError());
	check(cudaDeviceSynchronize());

}

/*-----------------------------------------------------------*/
/*              Repulsive force evaluation (GPU)             */
/*-----------------------------------------------------------*/
void CUDA_GPU::Kernels::tracker_gpu(Eigen::ArrayXf atoms_x, Eigen::ArrayXf atoms_y)
{

	//=======================================//
	//          Transfer values....          //
	//=======================================//

	//======================================================//
	// Copy position vectors to device

	Eigen::ArrayXf::Map(atoms_x_h, atoms_x.rows()) = atoms_x; // Map to x vector (host)
	Eigen::ArrayXf::Map(atoms_y_h, atoms_y.rows()) = atoms_y; // Map to y vector (host)

	check(cudaMemcpy(atoms_x_d, atoms_x_h, bytes_rows, cudaMemcpyHostToDevice));
	check(cudaMemcpy(atoms_y_d, atoms_y_h, bytes_rows, cudaMemcpyHostToDevice));

	//======================================================//
	// Matrix grids
	float h_t = 2 * floor((sqrt(1024 * ((float)N_rows / (float)N_cols))) / 2); // block height in threads (floor up to nearest even number)
	float w_t = 2 * floor((h_t * ((float)N_cols / (float)N_rows)) / 2); // block width in threads (floor up to nearest even number)

	int n_blocks_h(div_up(N_cols, w_t));
	int n_blocks_w(div_up(N_rows, h_t));

	dim3 blockSize = dim3(h_t, w_t);
	dim3 gridSize = dim3(n_blocks_h, n_blocks_w);

	//======================================================//
	// Pairwise distance and difference calculation

	CUDA_GPU::calc_tracking_matrix << <gridSize, blockSize >> > (G_d, G_track_d, atoms_x_d, atoms_y_d, N_grids, N_rows, N_cols);

	//======================================================//
	// Percentage covered (rowise matrix reduction by CUBLAS)

#ifndef CUBLAS_NDEBUG
	cublascheck(cublasSgemv(local_handle, CUBLAS_OP_N, N_rows, N_cols, &alpha, G_track_d, N_rows,
		d_ones_track, 1, &beta, p_d, 1)); // rowwise multiplication
#endif CUBLAS_NDEBUG
	//======================================================//

	check(cudaPeekAtLastError());
	check(cudaDeviceSynchronize());

}

/*-----------------------------------------------------------*/
/*					Force vectors (getter)				     */
/*-----------------------------------------------------------*/
void CUDA_GPU::Kernels::get_forces(Eigen::ArrayXf *force_x, Eigen::ArrayXf *force_y)
{
	check(cudaMemcpy(force_x_h, force_x_d, bytes, cudaMemcpyDeviceToHost)); // Copy forces x to device 
	check(cudaMemcpy(force_y_h, force_y_d, bytes, cudaMemcpyDeviceToHost)); // Copy forces y to device

	*force_x = Eigen::Map<Eigen::ArrayXf>(force_x_h, N_actual); // Map forces x to Eigen array 
	*force_y = Eigen::Map<Eigen::ArrayXf>(force_y_h, N_actual); // Map forces y to Eigen array
}

/*-----------------------------------------------------------*/
/*				 Percentage vector (getter)				     */
/*-----------------------------------------------------------*/
void CUDA_GPU::Kernels::get_p(Eigen::ArrayXf *p)
{
	//======================================================//
	// Copy percentage covered vector to host
	check(cudaMemcpy(p_h, p_d, bytes_rows, cudaMemcpyDeviceToHost)); // Copy percentage to device 

	*p = Eigen::Map<Eigen::ArrayXf>(p_h, N_rows); // Map percentage to Eigen array 
}

/*-----------------------------------------------------------*/
/*					 Tracking matrix (getter)			     */
/*-----------------------------------------------------------*/
void CUDA_GPU::Kernels::get_G(Eigen::ArrayXXf *G)
{
	//======================================================//
	// Copy ground covered matrix to host
	//Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > >(G_h, N_rows, N_cols) = *G;

	//for (int i(0); i < N_rows; ++i) {
	//	for (int j(0); j < N_cols; ++j) {
	//		//cout << "(" << i << "," << j << "): " << G_h[i + N_cols * j] << ", ";
	//		cout << G_h[j + N_cols * i] << ", ";
	//	}
	//	cout << endl;
	//}
	//cout << endl;

	//check(cudaMemcpy(G_d, G_h, N_rows * N_cols * sizeof(float), cudaMemcpyHostToDevice)); // Copy tracking matrix to device 
	
	check(cudaMemcpy(G_h, G_d, N_rows * N_cols * sizeof(float), cudaMemcpyDeviceToHost)); // Copy tracking matrix to device 

	//for (int i(0); i < N_rows; ++i) {
	//	for (int j(0); j < N_cols; ++j) {
	//		//cout << "(" << i << "," << j << "): " << G_h[i + N_cols * j] << ", ";
	//		cout << G_h[j + N_cols * i] << ", ";
	//	}
	//	cout << endl;
	//}
	//cout << endl;

	// Map rowwise format
	*G = Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor > >(G_h, N_rows, N_cols);

	//for (int i(0); i < N_rows; ++i) {
	//	for (int j(0); j < N_cols; ++j) {
	//		//cout << "(" << i << "," << j << "): " << G->block(i,j,1,1) << ", ";
	//		cout << G->block(i, j, 1, 1) << ", ";
	//	}
	//	cout << endl;
	//}
}

/*-----------------------------------------------------------*/
/*					 Tracking matrix (getter)			     */
/*-----------------------------------------------------------*/
void CUDA_GPU::Kernels::get_G_trace(Eigen::ArrayXXf *G_track)
{

	check(cudaMemcpy(G_track_h, G_track_d, N_rows * N_cols * sizeof(float), cudaMemcpyDeviceToHost)); // Copy tracking matrix to device 

	// Map rowwise format
	*G_track = Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor > >(G_track_h, N_rows, N_cols);

}


/*-----------------------------------------------------------*/
/*						  Destructor					     */
/*-----------------------------------------------------------*/
CUDA_GPU::Kernels::~Kernels() 
{
	///* Destroy all memory allocation pointers and free memory */
	check(cudaFree(atoms_x_d));
	check(cudaFreeHost(atoms_x_h));
	check(cudaFree(atoms_y_d));
	check(cudaFreeHost(atoms_y_h));

	check(cudaFree(diffs_x_d));
	check(cudaFree(diffs_y_d));

	check(cudaFree(force_x_d));
	check(cudaFreeHost(force_x_h));
	check(cudaFree(force_y_d));
	check(cudaFreeHost(force_y_h));

	/* Destroy all memory allocation pointers and free memory */
	check(cudaFree(G_d));
	check(cudaFreeHost(G_h));
	check(cudaFree(G_track_d));
	check(cudaFreeHost(G_track_h));

	check(cudaFree(p_d));
	check(cudaFreeHost(p_h));

	check(cudaFree(d_ones_force));
	check(cudaFree(d_ones_track));
	//cublascheck(cublasDestroy(handle)); // destroy cublas handle to avoid malloc errors

	/* Reset CUDA device (this is redundant) */
	check(cudaPeekAtLastError());
	//check(cudaDeviceReset());
}
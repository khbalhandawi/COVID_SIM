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
 \see    CUDA_functions.cuh
 */
#include <thrust/host_vector.h>
#include  "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <Eigen/Core>

#include "CUDA_functions.h"
#include "kernels.cuh"
#include "Utilities.cuh"

using namespace std;

/*-----------------------------------------------------------*/
/*             CUBLAS ERROR MESSAGES ENUMERATOR              */
/*-----------------------------------------------------------*/
static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
	switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";

	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";

	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "<unknown>";
}

/*-----------------------------------------------------------*/
/*                   CUBLAS ERROR CHECKING                   */
/*-----------------------------------------------------------*/
inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
	if (CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d, error: %s\nterminating!\n", __FILE__, __LINE__, \
			_cublasGetErrorEnum(err)); \
			assert(0); \
	}
}

/*-----------------------------------------------------------*/
/*               CUBLAS ERROR CHECKING (macro)               */
/*-----------------------------------------------------------*/
#define cublascheck(ans) { __cublasSafeCall((ans), __FILE__, __LINE__); }

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
/*              Repulsive force evaluation (GPU)             */
/*-----------------------------------------------------------*/
DLL_API void pairwise_gpu(Eigen::ArrayXf *force_x, Eigen::ArrayXf *force_y, Eigen::ArrayXf atoms_x, 
	Eigen::ArrayXf atoms_y, float SD_factor, int threads_per_block)
{

	const int N(atoms_x.rows()); // number of elements

	//=======================================//
	//          Transfer values....          //
	//=======================================//

	
	size_t bytes = N * sizeof(float); // Size, in bytes, of each vector

	float* atoms_x_h; // x vector (host)
	check(cudaMallocHost(&atoms_x_h, bytes));

	float* atoms_y_h; // y vector (host)
	check(cudaMallocHost(&atoms_y_h, bytes));

	Eigen::ArrayXf::Map(atoms_x_h, atoms_x.rows()) = atoms_x; // Map to x vector (host)
	Eigen::ArrayXf::Map(atoms_y_h, atoms_y.rows()) = atoms_y; // Map to y vector (host)

	float* atoms_x_d; // x vector (device)
	check(cudaMalloc(&atoms_x_d, bytes));

	float* atoms_y_d; // y vector (device)
	check(cudaMalloc(&atoms_y_d, bytes));

	check(cudaMemcpy(atoms_x_d, atoms_x_h, bytes, cudaMemcpyHostToDevice));
	check(cudaMemcpy(atoms_y_d, atoms_y_h, bytes, cudaMemcpyHostToDevice));

	//======================================================//
	// Matrix grids
	int n_blocks(div_up(N, sqrt(threads_per_block)));

	dim3 blockSize = dim3(sqrt(threads_per_block), sqrt(threads_per_block));
	dim3 gridSize = dim3(n_blocks, n_blocks);

	//======================================================//
	// Pairwise distance and difference calculation
	float* diffs_x_d;
	check(cudaMalloc(&diffs_x_d, N * bytes));

	float* diffs_y_d;
	check(cudaMalloc(&diffs_y_d, N * bytes));

	calc_force_m << <gridSize, blockSize >> > (diffs_x_d, diffs_y_d, atoms_x_d, atoms_y_d, SD_factor, N);
	//======================================================//
	// Force calculation (rowise matrix reduction by CUBLAS)

	float* force_x_d; // Force x vector (device)
	check(cudaMalloc(&force_x_d, bytes));

	float* force_y_d; // Force y vector (device)
	check(cudaMalloc(&force_y_d, bytes));

	float *d_ones; // vector of ones to multiply matrix with (device)
	check(cudaMalloc((void **)&d_ones, bytes));
	const float value = 1.f;
	initKernel << <n_blocks, threads_per_block >> > (d_ones, value, N); // initialize vector with ones using CUDA

	cublasHandle_t handle;
	cublascheck(cublasCreate(&handle)); // construct cublas handle

	float alpha = 1.f;
	float beta = 0.f;
	cublascheck(cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, thrust::raw_pointer_cast(diffs_x_d), N,
		thrust::raw_pointer_cast(d_ones), 1, &beta, thrust::raw_pointer_cast(force_x_d), 1)); // rowwise multiplication x
	cublascheck(cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, thrust::raw_pointer_cast(diffs_y_d), N,
		thrust::raw_pointer_cast(d_ones), 1, &beta, thrust::raw_pointer_cast(force_y_d), 1)); // rowwise multiplication y

	cublascheck(cublasDestroy(handle)); // destroy cublas handle to avoid malloc errors
	//======================================================//

	//=======================================//
	//          Retrieve values....          //
	//=======================================//

	check(cudaPeekAtLastError());
	check(cudaDeviceSynchronize());

	float* force_x_h; // Force x vector (device)
	check(cudaMallocHost(&force_x_h, bytes));

	float* force_y_h; // Force y vector (device)
	check(cudaMallocHost(&force_y_h, bytes));

	check(cudaMemcpy(force_x_h, force_x_d, bytes, cudaMemcpyDeviceToHost)); // Copy forces x to device 
	check(cudaMemcpy(force_y_h, force_y_d, bytes, cudaMemcpyDeviceToHost)); // Copy forces y to device

	*force_x = Eigen::Map<Eigen::ArrayXf>(force_x_h, N); // Map forces x to Eigen array 
	*force_y = Eigen::Map<Eigen::ArrayXf>(force_y_h, N); // Map forces y to Eigen array

	/* Destroy all memory allocation pointers and free memory */
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
}

/*-----------------------------------------------------------*/
/*              Repulsive force evaluation (GPU)             */
/*-----------------------------------------------------------*/
DLL_API void tracker_gpu(Eigen::ArrayXXf *G, Eigen::ArrayXf *p, Eigen::ArrayXf atoms_x, Eigen::ArrayXf atoms_y,
	const int n_pop, const int n_grids, const int threads_per_block)
{

	const int N_rows(n_pop); // number of rows
	const int N_cols(n_grids * n_grids); // number of rows

	//=======================================//
	//          Transfer values....          //
	//=======================================//

	//======================================================//
	// Copy position vectors to device
	size_t bytes_rows = N_rows * sizeof(float); // Size, in bytes, of each vector
	size_t bytes_cols = N_cols * sizeof(float); // Size, in bytes, of each vector

	float* atoms_x_h; // x vector (host)
	check(cudaMallocHost(&atoms_x_h, bytes_rows));

	float* atoms_y_h; // y vector (host)
	check(cudaMallocHost(&atoms_y_h, bytes_rows));

	Eigen::ArrayXf::Map(atoms_x_h, atoms_x.rows()) = atoms_x; // Map to x vector (host)
	Eigen::ArrayXf::Map(atoms_y_h, atoms_y.rows()) = atoms_y; // Map to y vector (host)

	float* atoms_x_d; // x vector (device)
	check(cudaMalloc(&atoms_x_d, bytes_rows));

	float* atoms_y_d; // y vector (device)
	check(cudaMalloc(&atoms_y_d, bytes_rows));

	check(cudaMemcpy(atoms_x_d, atoms_x_h, bytes_rows, cudaMemcpyHostToDevice));
	check(cudaMemcpy(atoms_y_d, atoms_y_h, bytes_rows, cudaMemcpyHostToDevice));

	//======================================================//
	// Copy ground covered matrix to device
	float* G_h; // Force x vector (device)
	check(cudaMallocHost(&G_h, N_rows * N_cols * sizeof(float)));
	Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > >(G_h, N_rows, N_cols) = *G;

	//for (int i(0); i < N_rows; ++i) {
	//	for (int j(0); j < N_cols; ++j) {
	//		//cout << "(" << i << "," << j << "): " << G_h[i + N_cols * j] << ", ";
	//		cout << G_h[j + N_cols * i] << ", ";
	//	}
	//	cout << endl;
	//}
	//cout << endl;

	float* G_d;
	check(cudaMalloc(&G_d, N_rows * N_cols * sizeof(float)));

	check(cudaMemcpy(G_d, G_h, N_rows * N_cols * sizeof(float), cudaMemcpyHostToDevice)); // Copy tracking matrix to device 

	//======================================================//
	// Matrix grids
	float h_t = 2 * floor((sqrt(1024 * ((float)N_rows / (float)N_cols))) / 2); // block height in threads (floor up to nearest even number)
	float w_t = 2 * floor((h_t * ((float)N_cols / (float)N_rows)) / 2); // block width in threads (floor up to nearest even number)

	int n_blocks_h(div_up(N_cols, w_t));
	int n_blocks_w(div_up(N_rows, h_t));

	dim3 blockSize = dim3(h_t, w_t);
	dim3 gridSize = dim3(n_blocks_h, n_blocks_w);
	//dim3 blockSize = dim3(1, 1);
	//dim3 gridSize = dim3(10, 9);

	//======================================================//
	// Pairwise distance and difference calculation

	calc_tracking_matrix << <gridSize, blockSize >> > (G_d, atoms_x_d, atoms_y_d, n_pop, n_grids, N_rows, N_cols);

	//======================================================//
	// Percentage covered (rowise matrix reduction by CUBLAS)

	float* p_d; // percentage vector (device)
	check(cudaMalloc(&p_d, bytes_rows));

	float *d_ones; // vector of ones to multiply matrix with (device)
	check(cudaMalloc((void **)&d_ones, bytes_cols));
	const float value = 1.f;

	int n_blocks(div_up(N_cols, sqrt(threads_per_block)));
	initKernel << <n_blocks, threads_per_block >> > (d_ones, value, N_cols); // initialize vector with ones using CUDA

	cublasHandle_t handle;
	cublascheck(cublasCreate(&handle)); // construct cublas handle

	float alpha = 1.f;
	float beta = 0.f;
	cublascheck(cublasSgemv(handle, CUBLAS_OP_T, N_cols, N_rows, &alpha, thrust::raw_pointer_cast(G_d), N_cols,
		thrust::raw_pointer_cast(d_ones), 1, &beta, thrust::raw_pointer_cast(p_d), 1)); // rowwise multiplication

	cublascheck(cublasDestroy(handle)); // destroy cublas handle to avoid malloc errors

	//=======================================//
	//          Retrieve values....          //
	//=======================================//

	//======================================================//
	// Copy percentage covered vector to host
	float* p_h; // percentage vector (device)
	check(cudaMallocHost(&p_h, bytes_rows));

	check(cudaMemcpy(p_h, p_d, bytes_rows, cudaMemcpyDeviceToHost)); // Copy percentage to device 

	*p = Eigen::Map<Eigen::ArrayXf>(p_h, N_rows); // Map percentage to Eigen array 

	//======================================================//
	// Copy ground covered matrix to host
	check(cudaPeekAtLastError());
	check(cudaDeviceSynchronize());

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
	*G = Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > >(G_h, N_rows, N_cols);

	//for (int i(0); i < N_rows; ++i) {
	//	for (int j(0); j < N_cols; ++j) {
	//		//cout << "(" << i << "," << j << "): " << G->block(i,j,1,1) << ", ";
	//		cout << G->block(i, j, 1, 1) << ", ";
	//	}
	//	cout << endl;
	//}

	/* Destroy all memory allocation pointers and free memory */
	check(cudaFree(atoms_x_d));
	check(cudaFreeHost(atoms_x_h));
	check(cudaFree(atoms_y_d));
	check(cudaFreeHost(atoms_y_h));

	check(cudaFree(G_d));
	check(cudaFreeHost(G_h));

	check(cudaFree(p_d));
	check(cudaFreeHost(p_h));
}

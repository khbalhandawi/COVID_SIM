#include  "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/iterator/discard_iterator.h>
#include <chrono>

#include <iostream>
#include <stdio.h>
#include <vector>
#include <random>
#include <Eigen/Core>

using namespace std;
#include "pairwise_dist.h"

/**************************************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX - NEEDED FOR APPROACH #1 */
/**************************************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T, T> {

	T Ncols; // --- Number of columns

	__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

	__host__ __device__ T operator()(T i) { return i / Ncols; }
};

#define check(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, char *file, int line)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

int div_up(int a, int b) {
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

//__global__ void calc_distances(float* distances, float* atoms_x, float* atoms_y, int N);
__global__ void calc_force_m(float* diffs_x, float* diffs_y, float* atoms_x, float* atoms_y, float SD_factor, int N);
__global__ void calc_forces(float* force, float* force_m, int N);

__declspec(dllexport) void pairwise_gpu(Eigen::ArrayXf *force_x, Eigen::ArrayXf *force_y, Eigen::ArrayXf atoms_x, Eigen::ArrayXf atoms_y, float SD_factor, int threads_per_block)
{
	
	const int N(atoms_x.rows());
	
	// Transfer values....
	
	// Size, in bytes, of each vector
	size_t bytes = N * sizeof(float);

	float* atoms_x_h;
	check(cudaMallocHost(&atoms_x_h, bytes));

	float* atoms_y_h;
	check(cudaMallocHost(&atoms_y_h, bytes));

	Eigen::ArrayXf::Map(atoms_x_h, atoms_x.rows()) = atoms_x;
	Eigen::ArrayXf::Map(atoms_y_h, atoms_y.rows()) = atoms_y;

	float* atoms_x_d;
	check(cudaMalloc(&atoms_x_d, bytes));

	float* atoms_y_d;
	check(cudaMalloc(&atoms_y_d, bytes));

	check(cudaMemcpy(atoms_x_d, atoms_x_h, bytes, cudaMemcpyHostToDevice));
	check(cudaMemcpy(atoms_y_d, atoms_y_h, bytes, cudaMemcpyHostToDevice));

	//======================================================//
	// Matrix grids
	int n_blocks(div_up(N, sqrt(threads_per_block)));

	dim3 blockSize = dim3(sqrt(threads_per_block), sqrt(threads_per_block));
	dim3 gridSize = dim3(n_blocks, n_blocks);

	//======================================================//
	// Distance calculation
	float* diffs_x_d;
	check(cudaMalloc(&diffs_x_d, N * bytes));

	float* diffs_y_d;
	check(cudaMalloc(&diffs_y_d, N * bytes));

	calc_force_m << <gridSize, blockSize >> > (diffs_x_d, diffs_y_d, atoms_x_d, atoms_y_d, SD_factor, N);
	//======================================================//
	// Force calculation
	
	// --- Matrix allocation and initialization
	thrust::device_vector<float> diffs_matrix_x_d(N * N);
	thrust::device_vector<float> diffs_matrix_y_d(N * N);

	diffs_matrix_x_d[0] = *diffs_x_d;
	diffs_matrix_y_d[0] = *diffs_y_d;

	// --- Allocate space for row sums and indices
	thrust::device_vector<float> force_x_d(N);
	thrust::device_vector<int> d_row_indices_x(N);
	thrust::device_vector<float> force_y_d(N);
	thrust::device_vector<int> d_row_indices_y(N);

	// --- Compute row sums by summing values with equal row indices
	//thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
	//                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
	//                    d_matrix.begin(),
	//                    d_row_indices.begin(),
	//                    d_row_sums.begin(),
	//                    thrust::equal_to<int>(),
	//                    thrust::plus<float>());

	thrust::reduce_by_key(
		thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(N)),
		thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(N)) + (N*N),
		diffs_matrix_x_d.begin(),
		thrust::make_discard_iterator(),
		force_x_d.begin());

	thrust::reduce_by_key(
		thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(N)),
		thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(N)) + (N*N),
		diffs_matrix_y_d.begin(),
		thrust::make_discard_iterator(),
		force_y_d.begin());

	//======================================================//

	// Retrieve values....

	check(cudaPeekAtLastError());
	check(cudaDeviceSynchronize());

	// --- Allocate space for row sums and indices on host
	thrust::host_vector<float> force_x_h(N);
	thrust::host_vector<float> force_y_h(N);

	force_x_h = force_x_d; force_y_h = force_y_d;

	*force_x = Eigen::Map<Eigen::ArrayXf>(force_x_h.data(), N);
	*force_y = Eigen::Map<Eigen::ArrayXf>(force_y_h.data(), N);

	check(cudaFree(atoms_x_d));
	check(cudaFreeHost(atoms_x_h));
	check(cudaFree(atoms_y_d));
	check(cudaFreeHost(atoms_y_h));

	check(cudaFree(diffs_x_d));
	check(cudaFree(diffs_y_d));
}

int main(int argc, char **argv)
{
	
	const int N(10);
	
	//auto seed = std::chrono::system_clock::now().time_since_epoch().count();//seed
	std::default_random_engine dre(0);//engine
	std::uniform_real_distribution<float> di(0.0, 1.0);//distribution

	uniform_real_distribution<float> distribution_ur(0.0, 1.0);
	auto distribution = [&](float) {return distribution_ur(dre); };

	Eigen::ArrayXf x = Eigen::ArrayXXf::NullaryExpr(N, 1, distribution);
	Eigen::ArrayXf y = Eigen::ArrayXXf::NullaryExpr(N, 1, distribution);
	Eigen::ArrayXf force_x(N), force_y(N);

	float SD_factor = 0.1;

	for (int i(0); i < x.rows(); i++) {
		cout << x[i] << ", " << y[i] << endl;
	}

	const int threads_per_block(32);

	cout << "n_blocks:" << div_up(N, threads_per_block) << endl;

	pairwise_gpu(&force_x, &force_y, x, y, SD_factor, threads_per_block);

	cout << "Result" << endl;
	for (int i(0); i < x.rows(); i++) {
		cout << force_x[i] << ", " << force_y[i] << endl;
	}

	return 0;
}

__global__ void calc_force_m(float* diffs_x, float* diffs_y, float* atoms_x, float* atoms_y, float SD_factor, int N)
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

__global__ void calc_forces(float* force, float* force_m, int N)
{
	int i(threadIdx.x + blockIdx.x * blockDim.x);

	if (i < N) {
		float sum = 0;
		for (int k = 0; k < N; k++) {
			sum += force_m[i*N + k];
		}

		force[i] = sum;
	}
}

//__global__ void calc_distances(float* force_x, float* force_y,
//	float* atoms_x, float* atoms_y, float social_distance_factor, int N)
//{
//	int i(threadIdx.x + blockIdx.x * blockDim.x);
//	int j(threadIdx.y + blockIdx.y * blockDim.y);
//
//	if (i >= N || j >= N) {
//		return;
//	}
//
//	//distances[i + N * j] =
//	//	(atoms_x[i] - atoms_x[j]) * (atoms_x[i] - atoms_x[j]) +
//	//	(atoms_y[i] - atoms_y[j]) * (atoms_y[i] - atoms_y[j]);
//}
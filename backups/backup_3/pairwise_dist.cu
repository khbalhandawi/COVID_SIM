#include  "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>

#include <iostream>
#include <stdio.h>
#include <vector>
#include <random>
#include <Eigen/Core>

using namespace std;
#include "pairwise_dist.h"

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

__global__ void calc_distances(float* distances, float* atoms_x, float* atoms_y, int N);

__declspec(dllexport) void pairwise_gpu(Eigen::ArrayXXf *distances, Eigen::ArrayXf atoms_x, Eigen::ArrayXf atoms_y, int threads_per_block)
{
	
	const int N(atoms_x.rows());
	
	// Transfer values....
	
	// Size, in bytes, of each vector
	size_t bytes = N * sizeof(float);

	float* atoms_x_h;
	check(cudaMallocHost(&atoms_x_h, bytes));

	float* atoms_y_h;
	check(cudaMallocHost(&atoms_y_h, bytes));

	//vector<float>& atoms_x_Ref = *atoms_x; // vector is not copied here
	//vector<float>& atoms_y_Ref = *atoms_y; // vector is not copied here

	for (int k(0); k < N; ++k) {
		atoms_x_h[k] = atoms_x[k];
		atoms_y_h[k] = atoms_y[k];
	}

	float* atoms_x_d;
	check(cudaMalloc(&atoms_x_d, bytes));

	float* atoms_y_d;
	check(cudaMalloc(&atoms_y_d, bytes));


	check(cudaMemcpy(atoms_x_d, atoms_x_h, bytes, cudaMemcpyHostToDevice));
	check(cudaMemcpy(atoms_y_d, atoms_y_h, bytes, cudaMemcpyHostToDevice));

	float* distances_d;
	check(cudaMalloc(&distances_d, N * bytes));

	int n_blocks(div_up(N, sqrt(threads_per_block)));

	dim3 blockSize = dim3(sqrt(threads_per_block), sqrt(threads_per_block));
	dim3 gridSize = dim3(n_blocks, n_blocks);

	calc_distances<<<gridSize, blockSize>>>(distances_d, atoms_x_d, atoms_y_d, N);

	// Retrieve values....

	check(cudaPeekAtLastError());
	check(cudaDeviceSynchronize());

	float* distances_h;
	check(cudaMallocHost(&distances_h, N * bytes));

	check(cudaMemcpy(distances_h, distances_d, N * bytes, cudaMemcpyDeviceToHost));

	//Eigen::ArrayXXf distances_ref = Eigen::Map<Eigen::ArrayXXf>(distances_h, N, N);
	*distances = Eigen::Map<Eigen::ArrayXXf>(distances_h, N, N);

	//for (int i(0); i < N; ++i) {
	//	for (int j(0); j < N; ++j) {
	//		distances_Ref[i][j] = distances_h[i + N * j];
	//		cout << "(" << i << "," << j << "): " << distances_h[i + N * j] << endl;
	//	}
	//}

	check(cudaFree(distances_d));
	check(cudaFreeHost(distances_h));
	check(cudaFree(atoms_x_d));
	check(cudaFreeHost(atoms_x_h));
	check(cudaFree(atoms_y_d));
	check(cudaFreeHost(atoms_y_h));
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
	Eigen::ArrayXXf d(N,N);

	for (int i(0); i < x.rows(); i++) {
		cout << x[i] << ", " << y[i] << endl;
	}

	const int threads_per_block(2);

	cout << "n_blocks:" << div_up(N, threads_per_block) << endl;

	pairwise_gpu(&d, x, y, threads_per_block);

	cout << "Result" << endl;
	cout << d << endl;

	return 0;
}

__global__ void calc_distances(float* distances,
	float* atoms_x, float* atoms_y, int N)
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
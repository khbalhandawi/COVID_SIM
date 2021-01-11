#include  "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>

#include <iostream>
#include <stdio.h>
#include <vector>
#include <random>
//#include <Eigen/Core>

using namespace std;
#include "pairwise_dist.cuh"



namespace GPUFuncs
{
	
	inline void GPUFuncs::_check(cudaError_t code, char *file, int line)
	{
		if (code != cudaSuccess) {
			fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
			exit(code);
		}
	}

	int GPUFuncs::div_up(int a, int b) {
		return ((a % b) != 0) ? (a / b + 1) : (a / b);
	}

	void GPUFuncs::check(cudaError_t code) {
		return _check((code), __FILE__, __LINE__);;
	}

	void GPUFuncs::pairwise_gpu(vector<vector<double>> *distances, vector<double> atoms_x, vector<double> atoms_y, int threads_per_block)
	{

		const int N(atoms_x.size());

		// Transfer values....

		// Size, in bytes, of each vector
		size_t bytes = N * sizeof(double);

		double* atoms_x_h;
		check(cudaMallocHost(&atoms_x_h, bytes));

		double* atoms_y_h;
		check(cudaMallocHost(&atoms_y_h, bytes));

		//vector<double>& atoms_x_Ref = *atoms_x; // vector is not copied here
		//vector<double>& atoms_y_Ref = *atoms_y; // vector is not copied here

		for (int k(0); k < N; ++k) {
			atoms_x_h[k] = atoms_x[k];
			atoms_y_h[k] = atoms_y[k];
		}

		double* atoms_x_d;
		check(cudaMalloc(&atoms_x_d, bytes));

		double* atoms_y_d;
		check(cudaMalloc(&atoms_y_d, bytes));


		check(cudaMemcpy(atoms_x_d, atoms_x_h, bytes, cudaMemcpyHostToDevice));
		check(cudaMemcpy(atoms_y_d, atoms_y_h, bytes, cudaMemcpyHostToDevice));

		double* distances_d;
		check(cudaMalloc(&distances_d, N * bytes));

		int n_blocks(div_up(N, sqrt(threads_per_block)));

		dim3 blockSize = dim3(sqrt(threads_per_block), sqrt(threads_per_block));
		dim3 gridSize = dim3(n_blocks, n_blocks);

		calc_distances << <gridSize, blockSize >> > (distances_d, atoms_x_d, atoms_y_d, N);

		// Retrieve values....

		check(cudaPeekAtLastError());
		check(cudaDeviceSynchronize());

		double* distances_h;
		check(cudaMallocHost(&distances_h, N * bytes));

		check(cudaMemcpy(distances_h, distances_d, N * bytes, cudaMemcpyDeviceToHost));

		vector<vector<double>>& distances_Ref = *distances; // vector is not copied here

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

	void GPUFuncs::calc_distances(double* distances,
		double* atoms_x, double* atoms_y, int N)
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
}

int main(int argc, char **argv)
{
	
	const int N(10);
	
	//auto seed = std::chrono::system_clock::now().time_since_epoch().count();//seed
	std::default_random_engine dre(0);//engine
	std::uniform_real_distribution<double> di(0.0, 1.0);//distribution

	std::vector<double> x(N), y(N);
	vector<vector<double>> d(N, vector<double>(N));
	std::generate(x.begin(), x.end(), [&] { return di(dre); });
	std::generate(y.begin(), y.end(), [&] { return di(dre); });

	for (int i(0); i < x.size(); i++) {
		cout << x[i] << ", " << y[i] << endl;
	}

	const int threads_per_block(2);

	GPUFuncs::GPUFuncs::pairwise_gpu(&d, x, y, threads_per_block);

	for (int i(0); i < N; ++i) {
		for (int j(0); j < N; ++j) {
			cout << d[i][j] << ", "; 
		}
		cout << endl;
	}

	return 0;
}
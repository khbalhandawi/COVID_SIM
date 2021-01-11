#pragma once

namespace GPUFuncs
{
	class GPUFuncs
	{
	public:

		// returns pairwise distances
		//extern "C" __declspec(dllexport) void pairwise_gpu(vector<vector<double>> *distances, vector<double> atoms_x, vector<double> atoms_y, int threads_per_block);
		static __declspec(dllexport) void pairwise_gpu(vector<vector<double>> *distances, vector<double> atoms_x, vector<double> atoms_y, int threads_per_block);

	private:
		static void calc_distances(double* distances, double* atoms_x, double* atoms_y, int N);
		static inline void _check(cudaError_t code, char *file, int line);
		static int div_up(int a, int b);
		static void check(cudaError_t code);
		//#define check(ans) { _check((ans), __FILE__, __LINE__); }
	};
}
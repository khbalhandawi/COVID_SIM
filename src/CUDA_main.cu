

#include <thrust/host_vector.h>
#include <chrono>

#include <iostream>
#include <stdio.h>
#include <vector>
#include <random>
#include <Eigen/Core>

#include  "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

using namespace std;

#include "CUDA_functions.h"
#include "kernels.cuh"
#include "Utilities.cuh"

/*-----------------------------------------------------------*/
/*                         MAIN void                         */
/*-----------------------------------------------------------*/
int main(int argc, char **argv)
{
	
	const int N(10);
	
	//auto seed = std::chrono::system_clock::now().time_since_epoch().count(); // random seed
	std::default_random_engine dre(0); //engine
	uniform_real_distribution<float> distribution_ur(0.0, 1.0);
	auto distribution = [&](float) {return distribution_ur(dre); }; // distribution lambda function

	Eigen::ArrayXf x = Eigen::ArrayXXf::NullaryExpr(N, 1, distribution); // Generate randomly distributed vector (x)
	Eigen::ArrayXf y = Eigen::ArrayXXf::NullaryExpr(N, 1, distribution); // Generate randomly distributed vector (y)
	Eigen::ArrayXf force_x(N), force_y(N); // Initialize force arrays

	float SD_factor = 0.1; // force amplitude (gravitational constant)

	if (N < 20) {
		for (int i(0); i < x.rows(); i++) {
			cout << x[i] << ", " << y[i] << endl;
		}
	}

	const int threads_per_block(1024); // number of thrreads
	cout << "n_blocks:" << div_up(N, sqrt(threads_per_block)) << endl;

	pairwise_gpu(&force_x, &force_y, x, y, SD_factor, threads_per_block); // Compute forces using GPU

	cout << "Result" << endl;
	if (N < 20) {
		for (int i(0); i < x.rows(); i++) {
			cout << force_x[i] << ", " << force_y[i] << endl;
		}
	}

	const int n_pop(10);
	const int n_grids(4);

	const int N_rows(n_pop); // number of rows
	const int N_cols(n_grids * n_grids); // number of rows

	Eigen::ArrayXf x_2(10), y_2(10);

	x_2 << 0.4800232, 0.00636118, 0.33891534, 0.07723257, 0.78516991, 0.06982263, 0.88362897, 0.84830657, 0.10674412, 0.69048652;
	y_2 << 0.26117482, 0.10601773, 0.35785163, 0.14943428, 0.23021565, 0.62097928, 0.69665474, 0.06754109, 0.61103961, 0.69224797;

	//x_2 << 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26;
	//y_2 << 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26;

	Eigen::ArrayXXf G(N_rows, N_cols);
	Eigen::ArrayXf p(N_rows); // Initialize percentage arrays

	G << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

	//G.block(0, 13, 1, 1) = 1;

	cout << G << endl;

	if (n_pop < 20) {
		for (int i(0); i < x_2.rows(); i++) {
			cout << x_2[i] << ", " << y_2[i] << endl;
		}
	}

	tracker_gpu(&G, &p, x_2, y_2, n_pop, n_grids, threads_per_block);

	cout << G << endl;
	cout << p << endl;

	return 0;
}
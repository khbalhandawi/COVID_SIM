

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
#ifndef RANDOM_DEBUG
	const int n_pop(10);
	const int n_grids(4);
#else
	const int n_pop(20);
	const int n_grids(4);
#endif
	
	//auto seed = std::chrono::system_clock::now().time_since_epoch().count(); // random seed
	std::default_random_engine dre(0); //engine
	uniform_real_distribution<float> distribution_ur(0.0, 1.0);
	auto distribution = [&](float) {return distribution_ur(dre); }; // distribution lambda function

	Eigen::ArrayXf x = Eigen::ArrayXXf::NullaryExpr(n_pop, 1, distribution); // Generate randomly distributed vector (x)
	Eigen::ArrayXf y = Eigen::ArrayXXf::NullaryExpr(n_pop, 1, distribution); // Generate randomly distributed vector (y)
	Eigen::ArrayXf force_x(n_pop), force_y(n_pop); // Initialize force arrays

	float SD_factor = 0.1; // force amplitude (gravitational constant)

	if (n_pop <= 20) {
		for (int i(0); i < x.rows(); i++) {
			cout << x[i] << ", " << y[i] << endl;
		}
	}

	const int threads_per_block(1024); // number of thrreads
	cout << "n_blocks:" << div_up(n_pop, sqrt(threads_per_block)) << endl;

	CUDA_GPU::Kernels ABM_cuda(n_pop, n_grids, threads_per_block);

	ABM_cuda.pairwise_gpu(x, y, SD_factor); // Compute forces using GPU
	ABM_cuda.get_forces(&force_x, &force_y); // get GPU forces

	cout << "Result" << endl;
	if (n_pop <= 20) {
		for (int i(0); i < x.rows(); i++) {
			cout << force_x[i] << ", " << force_y[i] << endl;
		}
	}

	x = Eigen::ArrayXXf::NullaryExpr(n_pop - 8, 1, distribution); // Generate randomly distributed vector (x)
	y = Eigen::ArrayXXf::NullaryExpr(n_pop - 8, 1, distribution); // Generate randomly distributed vector (y)
	Eigen::ArrayXf force_x2(n_pop - 8), force_y2(n_pop - 8); // Initialize force arrays

	if (n_pop <= 20) {
		for (int i(0); i < x.rows(); i++) {
			cout << x[i] << ", " << y[i] << endl;
		}
	}

	ABM_cuda.pairwise_gpu(x, y, SD_factor); // Compute forces using GPU
	ABM_cuda.get_forces(&force_x2, &force_y2); // get GPU forces

	cout << "Result" << endl;
	if (n_pop <= 20) {
		for (int i(0); i < x.rows(); i++) {
			cout << force_x2[i] << ", " << force_y2[i] << endl;
		}
	}

#ifndef RANDOM_DEBUG
	const int N_rows(n_pop); // number of rows
	const int N_cols(n_grids * n_grids); // number of rows

	//Eigen::ArrayXf x_2(10), y_2(10);
	Eigen::ArrayXf x_2(10), y_2(10);

	x_2 << 0.4800232, 0.00636118, 0.33891534, 0.07723257, 0.78516991, 0.06982263, 0.88362897, 0.84830657, 0.10674412, 0.69048652;
	y_2 << 0.26117482, 0.10601773, 0.35785163, 0.14943428, 0.23021565, 0.62097928, 0.69665474, 0.06754109, 0.61103961, 0.69224797;
	//x_2 << 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26;
	//y_2 << 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26;

	Eigen::ArrayXXf G(N_rows, N_cols);
	Eigen::ArrayXf p(N_rows); // Initialize percentage arrays

	G << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
#else

	const int N_rows(n_pop); // number of rows
	const int N_cols(n_grids * n_grids); // number of rows

	Eigen::ArrayXf x_2 = Eigen::ArrayXXf::NullaryExpr(n_pop, 1, distribution); // Generate randomly distributed vector (x)
	Eigen::ArrayXf y_2 = Eigen::ArrayXXf::NullaryExpr(n_pop, 1, distribution); // Generate randomly distributed vector (y)

	Eigen::ArrayXXf G(N_rows, N_cols);
	Eigen::ArrayXf p(N_rows); // Initialize percentage arrays
#endif

	if (n_pop <= 20) {
		for (int i(0); i < x_2.rows(); i++) {
			cout << x_2[i] << ", " << y_2[i] << endl;
		}
	}

	ABM_cuda.tracker_gpu(x_2, y_2);
	ABM_cuda.get_p(&p);
	ABM_cuda.get_G_trace(&G);

	cout << G << endl;
	cout << p << endl;

#ifndef RANDOM_DEBUG
	x_2 << 0.0800232, 0.00636118, 0.33891534, 0.07723257, 0.78516991, 0.06982263, 0.88362897, 0.84830657, -1.00000000, -1.00000000;
	y_2 << 0.9117482, 0.10601773, 0.35785163, 0.14943428, 0.23021565, 0.62097928, 0.69665474, 0.06754109, -1.00000000, -1.00000000;
#else
	x_2 = Eigen::ArrayXXf::NullaryExpr(n_pop, 1, distribution); // Generate randomly distributed vector (x)
	y_2 = Eigen::ArrayXXf::NullaryExpr(n_pop, 1, distribution); // Generate randomly distributed vector (y)
#endif

	if (n_pop <= 20) {
		for (int i(0); i < x_2.rows(); i++) {
			cout << x_2[i] << ", " << y_2[i] << endl;
		}
	}

	ABM_cuda.tracker_gpu(x_2, y_2);
	ABM_cuda.get_p(&p);
	ABM_cuda.get_G_trace(&G);

	cout << G << endl;
	cout << p << endl;

	return 0;
}
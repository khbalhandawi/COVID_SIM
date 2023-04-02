#pragma once

#include "Configuration.h"
#include "RandomDevice.h"
#include "Defines.h"

#ifdef GPU_ACC
#include "CUDA_functions.cuh"
#endif

#include<vector>
#include<string>
#include<Eigen/Core>

#ifndef POPULATION_TRACKERS_H_H
#define POPULATION_TRACKERS_H_H

namespace COVID_SIM {

API_BEGIN
	class DLL_API Population_trackers
	{
	public:

		std::vector<int> susceptible;
		std::vector<int> infectious;
		std::vector<int> recovered;
		std::vector<int> fatalities;
		Configuration Config;
		std::vector<double> distance_travelled;
		Eigen::ArrayXf total_distance; // distance travelled by individuals
		std::vector<double> mean_perentage_covered;
		std::vector<double> mean_R0; // mean basic reproductive number
		Eigen::ArrayXXf grid_coords;
		Eigen::ArrayXXf ground_covered;
		Eigen::ArrayXf perentage_covered; // portion of world covered by individuals
		// PLACEHOLDER - whether recovered individual can be reinfected
		bool reinfect;

#ifdef GPU_ACC
		/*-----------------------------------------------------------*/
		/*                   Update counts (CUDA)                    */
		/*-----------------------------------------------------------*/
		void update_counts_cuda(Eigen::ArrayXXf population, int frame, CUDA_GPU::Kernels *ABM_cuda);
#else
		/*-----------------------------------------------------------*/
		/*                      Update counts                        */
		/*-----------------------------------------------------------*/
		void update_counts(Eigen::ArrayXXf population, int frame);
#endif
		/*-----------------------------------------------------------*/
		/*                       Constructor                         */
		/*-----------------------------------------------------------*/
		Population_trackers(Configuration Config);

		/*-----------------------------------------------------------*/
		/*                        Destructor                         */
		/*-----------------------------------------------------------*/
		~Population_trackers();
	};
API_END

	/*-----------------------------------------------------------*/
	/*                  initialize population                    */
	/*-----------------------------------------------------------*/
	Eigen::ArrayXXf initialize_population(Configuration Config, RandomDevice *my_rand, int mean_age = 45, int max_age = 105,
		std::vector<double> xbounds = { 0, 1 }, std::vector<double> ybounds = { 0, 1 });

	/*-----------------------------------------------------------*/
	/*                 initialize destinations                   */
	/*-----------------------------------------------------------*/
	Eigen::ArrayXXf initialize_destination_matrix(int pop_size, int total_destinations,
		std::vector<double> destination_lower_bounds, std::vector<double> destination_upper_bounds);

	/*-----------------------------------------------------------*/
	/*                initialize ground covered                  */
	/*-----------------------------------------------------------*/
	void initialize_ground_covered_matrix(Eigen::ArrayXXf &grid_coords, Eigen::ArrayXXf &ground_covered, int pop_size, int n_gridpoints, std::vector<double> xbounds = { 0, 1 },
		std::vector<double> ybounds = { 0, 1 });

API_BEGIN
	/*-----------------------------------------------------------*/
	/*                   save population data                    */
	/*-----------------------------------------------------------*/
	void DLL_API save_data(Eigen::ArrayXXf population, Population_trackers pop_tracker, Configuration Config, int frame, std::string folder);
API_END

	/*-----------------------------------------------------------*/
	/*         save population data at current time step         */
	/*-----------------------------------------------------------*/
	void save_population(Eigen::ArrayXXf population, int tstep = 0, std::string folder = "data_tstep");

	/*-----------------------------------------------------------*/
	/*    save population ground covered at current time step    */
	/*-----------------------------------------------------------*/
	void save_ground_covered(Eigen::ArrayXXf ground_covered, int tstep = 0, std::string folder = "data_tstep");

API_BEGIN
	/*-----------------------------------------------------------*/
	/*                   save grid coordinates                   */
	/*-----------------------------------------------------------*/
	void DLL_API save_grid_coords(Eigen::ArrayXXf grid_coords, std::string folder = "data_tstep");
API_END

}

#endif // POPULATION_TRACKERS_H_H

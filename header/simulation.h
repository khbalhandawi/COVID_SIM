#pragma once

#include <Eigen/core>
#include "Configuration.h"
#include "Population_trackers.h"
#include "utilities.h"
#include "motion.h"
#include "RandomDevice.h"
#include "infection.h"
#include "tic_toc.h"
#ifdef GPU_ACC
#include "CUDA_functions.h"
#endif
#ifndef _N_QT
#include "visualizer.h"
#endif

#ifdef GPU_ACC
class simulation : Population_trackers, RandomDevice, CUDA_GPU::Kernels
#else
class simulation : Population_trackers, RandomDevice
#endif
{
public:
	Configuration Config;
	Population_trackers pop_tracker;
	RandomDevice my_rand;
#ifdef GPU_ACC
	CUDA_GPU::Kernels ABM_cuda;
#endif

	TicToc tc;
	// initialize times
	int frame;
	double time;
	double computation_time;
	double last_step_change;
	bool above_act_thresh;
	bool above_deact_thresh;
	bool above_test_thresh;
	// initialize default population
	Eigen::ArrayXXf population;
	// initialize pairwise distance matrix
	Eigen::ArrayXXf dist;
	// initalise destinations vector
	vector<double> lb_environments, ub_environments;
	Eigen::ArrayXXf destinations;
	// initalise grid for tracking population positions
	Eigen::ArrayXXf grid_coords;
	Eigen::ArrayXXf ground_covered;
	// RNG
	unsigned long seed;
#ifndef _N_QT
	// Visualization
	visualizer vis;
#endif
	// Ppopulation segments
	Eigen::ArrayXXf outside_world; // outside main world
	Eigen::ArrayXXf inside_world; // inside main world
	Eigen::ArrayXXf travelling_pop; // travelling individuals
	Eigen::ArrayXXf at_destination; // arrived individuals
	Eigen::ArrayXXf pop_infected; // infected individuals
	Eigen::ArrayXXf pop_hospitalized; // hospitalized individuals
	/*-----------------------------------------------------------*/
	/*                 (Re)initialize population                 */
	/*-----------------------------------------------------------*/
	void population_init();

	/*-----------------------------------------------------------*/
	/*                   Initialize simulation                   */
	/*-----------------------------------------------------------*/
	void initialize_simulation();

	/*-----------------------------------------------------------*/
	/*                         Time step                         */
	/*-----------------------------------------------------------*/
	void tstep();

	/*-----------------------------------------------------------*/
	/*                   Placeholder function                    */
	/*-----------------------------------------------------------*/
	void callback();

	/*-----------------------------------------------------------*/
	/*                       Run simulation                      */
	/*-----------------------------------------------------------*/
	void run();

	/*-----------------------------------------------------------*/
	/*                        Constructor                        */
	/*-----------------------------------------------------------*/
	simulation(Configuration Config_init, unsigned long seed);

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~simulation();
};


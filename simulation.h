#pragma once

#include <Eigen/core>
#include "Configuration.h"
#include "Population_trackers.h"
#include "utilities.h"
#include "motion.h"
#include "RandomDevice.h"
#include "visualizer.h"
#include "infection.h"
#include "tic_toc.h"

class simulation : Population_trackers, RandomDevice
{
public:
	Configuration Config;
	Population_trackers pop_tracker;
	RandomDevice my_rand;
	TicToc tc;
	// initialize times
	int frame;
	double time;
	int last_step_change;
	bool above_act_thresh;
	bool above_deact_thresh;
	bool above_test_thresh;
	// initialize default population
	Eigen::ArrayXXd population;
	// initalise destinations vector
	Eigen::ArrayXXd destinations;
	// initalise grid for tracking population positions
	Eigen::ArrayXXd grid_coords;
	Eigen::ArrayXXd ground_covered;
	// RNG
	unsigned long seed;
	// Visualization
	visualizer vis;
	// Ppopulation segments
	Eigen::ArrayXXd outside_world; // outside main world
	Eigen::ArrayXXd inside_world; // inside main world
	Eigen::ArrayXXd travelling_pop; // travelling individuals
	Eigen::ArrayXXd at_destination; // arrived individuals
	Eigen::ArrayXXd pop_infected; // infected individuals
	Eigen::ArrayXXd pop_hospitalized; // hospitalized individuals
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
	simulation(Configuration Config, unsigned long seed);

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~simulation();
};


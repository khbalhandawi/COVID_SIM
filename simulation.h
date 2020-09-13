#pragma once

#include <Eigen/core>
#include "Configuration.h"
#include "Population_trackers.h"
#include "utilities.h"
#include "motion.h"
#include "RandomDevice.h"
#include "visualizer.h"
#include "infection.h"

class simulation : Population_trackers, RandomDevice
{
public:
	Configuration Config;
	Population_trackers pop_tracker;
	RandomDevice my_rand;
	// initialize times
	int frame;
	double time;
	int last_step_change;
	bool above_act_thresh;
	bool above_deact_thresh;
	bool above_test_thresh;
	// initialize default population
	Eigen::ArrayXXf population;
	// initalise destinations vector
	Eigen::ArrayXXf destinations;
	// initalise grid for tracking population positions
	Eigen::ArrayXXf grid_coords;
	Eigen::ArrayXXf ground_covered;
	// RNG
	unsigned long seed;
	// Visualization
	visualizer vis;
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
	simulation(Configuration Config, unsigned long seed);

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~simulation();
};


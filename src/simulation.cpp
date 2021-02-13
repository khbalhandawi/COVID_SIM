#include "simulation.h"
#include "utilities.h"
#include "motion.h"
#include "infection.h"
#include "path_planning.h"

#include <iostream>

#ifdef GPU_ACC
COVID_SIM::simulation::simulation(Configuration Config_init, unsigned long seed) :
	Population_trackers(Config_init), RandomDevice(seed), CUDA_GPU::Kernels(Config_init.pop_size, Config_init.n_gridpoints - 1, 1024),
	pop_tracker(Config_init), my_rand(seed), ABM_cuda(Config_init.pop_size, Config_init.n_gridpoints - 1, 1024)
#else
COVID_SIM::simulation::simulation(Configuration Config_init, unsigned long seed) :
	Population_trackers(Config_init), RandomDevice(seed), 
	pop_tracker(Config_init), my_rand(seed)
#endif
{

	Config = Config_init;
	// initialize simulation
	frame = 0;

	// If user specified seed should be used
	if (Config.constant_seed) {
		my_rand.load_state();
		std::cout << "random_check: " << my_rand.rand() << std::endl;
	}

	// Initialize base classes
	population_init();
	initialize_simulation();

}

/*-----------------------------------------------------------*/
/*                 (Re)initialize population                 */
/*-----------------------------------------------------------*/
void COVID_SIM::simulation::population_init()
{
	/* (re-)initializes population */
	population = initialize_population(Config, &my_rand, Config.mean_age, Config.max_age, Config.xbounds, Config.ybounds);
}

/*-----------------------------------------------------------*/
/*                   Initialize simulation                   */
/*-----------------------------------------------------------*/
void COVID_SIM::simulation::initialize_simulation()
{
	/* initializes simulation */

	// initialize times
	frame = 0;
	time = 0;
	computation_time = 0.0; // in ms
	last_step_change = 0.0;
	above_act_thresh = false;
	above_deact_thresh = false;
	above_test_thresh = false;

	// initialize default population
	population_init();
	
	// initialize destinations vector
	lb_environments = { Config.xbounds[0], Config.ybounds[0], Config.isolation_bounds[0], Config.isolation_bounds[1] };
	ub_environments = { Config.xbounds[1], Config.ybounds[1], Config.isolation_bounds[2], Config.isolation_bounds[3] };
	destinations = initialize_destination_matrix(Config.pop_size, 2, lb_environments, ub_environments); // main world and quarantine world

	// initialize grid for tracking population positions
	initialize_ground_covered_matrix(grid_coords, ground_covered, Config.pop_size, Config.n_gridpoints, Config.xbounds, Config.ybounds);
	pop_tracker.grid_coords = grid_coords;
	pop_tracker.ground_covered = ground_covered;
}

/*-----------------------------------------------------------*/
/*                         Time step                         */
/*-----------------------------------------------------------*/
void COVID_SIM::simulation::tstep()
{
	/*
	takes a time step in the simulation
	*/
	
	Eigen::ArrayXXf outside_world = population(select_rows(population.col(11) != 0), Eigen::all); // outside main world
	Eigen::ArrayXXf inside_world = population(select_rows(population.col(11) == 0), Eigen::all); // inside main world
	Eigen::ArrayXXf travelling_pop = population(select_rows(population.col(12) == 0), Eigen::all); // travelling individuals
	Eigen::ArrayXXf at_destination = population(select_rows(population.col(12) == 1), Eigen::all); // arrived individuals
	Eigen::ArrayXXf pop_hospitalized = population(select_rows(population.col(10) == 1), Eigen::all); // hospitalized individuals
	Eigen::ArrayXXf pop_infected = population(select_rows(population.col(6) == 1), Eigen::all); // infected individuals
	tc.tic(); // reset clock
	//======================================================================================//
	// check destinations if active
	// define motion vectors if destinations active and not everybody is at destination

	if (travelling_pop.rows() > 0) {
		set_destination(population, destinations);
		check_at_destination(population, destinations, Config.wander_factor_dest);
	}

	if (at_destination.rows() > 0) {
		//keep them at destination
		keep_at_destination(population, lb_environments, ub_environments, Config.wall_buffer, Config.bounce_buffer);
	}
	//======================================================================================//
	//gravity wells
	if (Config.gravity_strength > 0.001) {
		update_gravity_forces(population, time, last_step_change, &my_rand, 
			Config.wander_step_size, Config.gravity_strength, 
			Config.wander_step_duration);
	}
	//======================================================================================//
	//activate social distancing above a certain infection threshold

	if (!(above_act_thresh) && (Config.social_distance_threshold_on > 0)) {
		// If not previously above infection threshold activate when threshold reached
		if (Config.thresh_type == "hospitalized") {
			above_act_thresh = select_rows(population.col(11) == 1).size() >= Config.social_distance_threshold_on;
		}
		else if (Config.thresh_type == "infected") {
			above_act_thresh = select_rows(population.col(6) == 1).size() >= Config.social_distance_threshold_on;
		}
			
	}
	else if (Config.social_distance_threshold_on == 0) {
		above_act_thresh = true;
	}

	//deactivate social distancing after infection drops below threshold after using social distancing
	if ((above_act_thresh) && !(above_deact_thresh) && (Config.social_distance_threshold_off > 0)) {
		// If previously went above infection threshold deactivate when threshold reached
		ArrayXXb cond(Config.pop_size, 2);
		cond << (population.col(6) == 1) , (population.col(11) == 0);
		above_deact_thresh = population(select_rows(cond), Eigen::all).rows() <= Config.social_distance_threshold_off;
	}

	bool act_social_distancing;

	// activate social distancing at the onset of infection
	if (!Config.SD_act_onset) {
		act_social_distancing = ((above_act_thresh) && !(above_deact_thresh) && (pop_infected.rows() > 0));
	}
	// activate social distancing from start of simulation
	else if (Config.SD_act_onset) {
		act_social_distancing = ((above_act_thresh) && !(above_deact_thresh));
	}

	//activate social distancing only for compliant individuals
	if ((Config.social_distance_factor > 0) && (act_social_distancing)) {

#ifdef GPU_ACC
		update_repulsive_forces_cuda(population, Config.social_distance_factor, &ABM_cuda);
#else
		update_repulsive_forces(population, Config.social_distance_factor, dist);
#endif // GPU_ACC
	
	}

	if (population.col(1).isNaN().any()) {
		std::cout << "================================" << std::endl;
		std::cout << "Saving random seed state" << std::endl;
		my_rand.save_state();
		std::cout << "random_check: " << my_rand.rand() << std::endl;
		std::cout << "Division by zero condition!" << std::endl;
		throw "Division by zero condition!";
	}

	//======================================================================================//
	//update velocities

	update_velocities(population, Config.max_speed, Config.dt);

	//for dead ones : set velocity and social distancing to 0 for dead ones
	population(select_rows(population.col(6) == 3), { 3,4 }) = 0;
	population(select_rows(population.col(6) == 3), { 17 }) = 1;

	//update positions
	update_positions(population, Config.dt);

	//======================================================================================//
	//find new infections

	if (!(above_test_thresh) && (Config.testing_threshold_on > 0)) {
		// If not previously above infection threshold activate when threshold reached
		above_test_thresh = (pop_infected.rows() >= Config.testing_threshold_on);
		//above_test_thresh = (pop_infected.rows() >= Config.social_distance_threshold_on);
	}
	else if (Config.testing_threshold_on == 0) {
		above_test_thresh = true;
	}

	bool act_testing = (above_test_thresh) && (pop_infected.rows() > 0);

	// Find infections and send to hospital if applicable
	infect(population, destinations, Config, frame, &my_rand, Config.self_isolate, 1, act_testing);

	// recover and die
	recover_or_die(population, destinations, Config, frame, &my_rand, 0);

	//======================================================================================//
	//update population statistics
#ifdef GPU_ACC
	pop_tracker.update_counts_cuda(population, frame, &ABM_cuda);
#else
	pop_tracker.update_counts(population, frame);
#endif // GPU_ACC


	//======================================================================================//
	//report stuff to console
	if ((Config.verbose) && ((frame % Config.report_freq) == 0)) {
		std::cout << frame;
		std::cout << ": S: " << pop_tracker.susceptible.back();
		std::cout << ": I: " << pop_tracker.infectious.back();
		std::cout << ": R: " << pop_tracker.recovered.back();
		std::cout << ": in treatment: " << pop_hospitalized.rows();
		std::cout << ": F: " << pop_tracker.fatalities.back();
		std::cout << ": of total: " << Config.pop_size;
		if (Config.track_position) {
			std::cout << ": D: " << pop_tracker.distance_travelled.back()*100;
		}
		if (Config.track_GC) {
			std::cout << ": GC: " << pop_tracker.mean_perentage_covered.back()*100;
		}
		if (Config.track_R0) {
			std::cout << ": R0: " << pop_tracker.mean_R0.back();
		}
		std::cout << " time: " << tc.toc() << " ms";
		std::cout << std::endl;
	}

	//save popdata if required
	if ((Config.save_pop) && ((frame % Config.save_pop_freq) == 0)) {
		save_population(population, frame, Config.save_pop_folder);
	}

	//save ground_covered if required
	if ((Config.save_ground_covered) && ((frame % Config.save_pop_freq) == 0)) {
		save_ground_covered(pop_tracker.ground_covered(0,Eigen::all), frame, Config.save_pop_folder);
	}

	//run callback
	callback();

	//======================================================================================//
	//update frame
	frame += 1;
	time += Config.dt;
	computation_time = tc.toc();
}

/*-----------------------------------------------------------*/
/*                   Placeholder function                    */
/*-----------------------------------------------------------*/
void COVID_SIM::simulation::callback()
{
	/*placeholder function that can be overwritten.

	By overwriting this method any custom behavior can be implemented.
	The method is called after every simulation timestep.
	*/

	if (frame == 50) {
		std::cout << "infecting person (Patient Zero)" << std::endl;

		if (Config.patient_Z_loc == "random") {
			population(0, 6) = 1;
			population(0, 8) = 50;
			population(0, 10) = 0; // do not place in treatment
		}
		else if (Config.patient_Z_loc == "central") {

			Eigen::ArrayXXf center = Eigen::ArrayXXf::Zero(Config.pop_size, 2);

			center.col(0) = (Config.xbounds[0] + Config.xbounds[1]) / 2;
			center.col(1) = (Config.ybounds[0] + Config.ybounds[1]) / 2;

			Eigen::ArrayXXf to_center = (center - population(Eigen::all, { 1,2 }));
			Eigen::ArrayXf dist = to_center.rowwise().norm().array();

			// infect nearest individual to center
			Eigen::ArrayXXf::Index minRow;
			dist.minCoeff(&minRow);

			population(minRow, 6) = 1;
			population(minRow, 8) = 50;
			population(minRow, 10) = 0; // do not place in treatment
		}

	}
		
}

/*-----------------------------------------------------------*/
/*                       Run simulation                      */
/*-----------------------------------------------------------*/
void COVID_SIM::simulation::run()
{
	/* run simulation */
	//save grid_coords if required
	if (Config.save_ground_covered) {
		save_grid_coords(pop_tracker.grid_coords, Config.save_pop_folder);
	}

	int i = 0;

	ArrayXXb cond(Config.pop_size, 2);

	while (i < Config.simulation_steps) {

		try
		{
			tstep(); // code that could cause exception

		}
		catch (const std::exception &exc)
		{
			// catch anything thrown within try block that derives from std::exception
			std::cerr << exc.what();
		}

		// check whether to end if no infectious persons remain.
		// check if frame is above some threshold to prevent early breaking when simulation
		// starts initially with no infections.
		if ((Config.endif_no_infections) && (frame >= 300)) {

			cond << (population.col(6) == 1), (population.col(6) == 4);
			if (population(select_rows_any(cond), Eigen::all).rows() == 0) {
				i = Config.simulation_steps;
			}
		}
		else {
			i += 1;
		}
	}

	if (Config.save_data) {
		save_data(population, pop_tracker, Config, (frame - 1), Config.save_pop_folder);
	}

	// report outcomes
	if (Config.verbose) {

		cond << (population.col(6) == 1), (population.col(6) == 4);

		Eigen::ArrayXXf pop_susceptible = population(select_rows(population.col(6) == 0), Eigen::all); // healthy individuals
		Eigen::ArrayXXf pop_recovered = population(select_rows(population.col(6) == 2), Eigen::all); // recovered individuals
		Eigen::ArrayXXf pop_fatality = population(select_rows(population.col(6) == 3), Eigen::all); // dead individuals
		Eigen::ArrayXXf pop_asymptomatic = population(select_rows(population.col(6) == 4), Eigen::all); // asymptomatic individuals
		Eigen::ArrayXXf pop_infectious = population(select_rows_any(cond), Eigen::all); // infectious individuals


		std::cout << "\n\n" << "-----stopping-----" << std::endl;
		std::cout << "total timesteps taken: " << std::to_string(frame) << std::endl;
		std::cout << "total fatalities: " << std::to_string(pop_fatality.rows()) << std::endl;
		std::cout << "total recovered: " << std::to_string(pop_recovered.rows()) << std::endl;
		std::cout << "total infected: " << std::to_string(pop_infected.rows()) << std::endl;
		std::cout << "total infectious: " << std::to_string(pop_infectious.rows()) << std::endl;
		std::cout << "total unaffected: " << std::to_string(pop_susceptible.rows()) << std::endl;
		if (Config.track_GC) {
			std::cout << "Mean % explored: " << pop_tracker.mean_perentage_covered.back()*100 << std::endl;
		}
		if (Config.track_R0) {
			std::cout << "Max R0: " << *max_element(pop_tracker.mean_R0.begin(), pop_tracker.mean_R0.end()) << std::endl;
		}
	}

}

/*-----------------------------------------------------------*/
/*                        Destructor                         */
/*-----------------------------------------------------------*/
COVID_SIM::simulation::~simulation()
{
}

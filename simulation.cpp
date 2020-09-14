#include "simulation.h"

simulation::simulation(Configuration Config_init, unsigned long seed) :
	Population_trackers(Config_init, grid_coords, ground_covered),
	RandomDevice(seed),
	pop_tracker(Config_init, grid_coords, ground_covered), my_rand(seed)
{

	Config = Config_init;
	// initialize simulation
	frame = 0;
	// load visualizer
	vis = visualizer();

	// If user specified seed should be used
	if (Config.constant_seed) {
		my_rand.load_state();
		cout << "random_check: " << my_rand.rand() << endl;
	}

	// Initialize base classes
	population_init();
	initialize_simulation();

}

/*-----------------------------------------------------------*/
/*                 (Re)initialize population                 */
/*-----------------------------------------------------------*/
void simulation::population_init()
{
	/* (re-)initializes population */
	population = initialize_population(Config, &my_rand, Config.mean_age, Config.max_age, Config.xbounds, Config.ybounds);
}

/*-----------------------------------------------------------*/
/*                   Initialize simulation                   */
/*-----------------------------------------------------------*/
void simulation::initialize_simulation()
{
	/* initializes simulation */

	// initialize times
	frame = 0;
	time = 0;
	last_step_change = 0;
	above_act_thresh = false;
	above_deact_thresh = false;
	above_test_thresh = false;

	// initialize default population
	population_init();
	
	// initialize destinations vector
	destinations = initialize_destination_matrix(Config.pop_size, 1);

	// initialize grid for tracking population positions
	tie(grid_coords, ground_covered) = initialize_ground_covered_matrix(Config.pop_size, Config.n_gridpoints, Config.xbounds, Config.ybounds);
	pop_tracker.grid_coords = grid_coords;
	pop_tracker.ground_covered = ground_covered;
}

/*-----------------------------------------------------------*/
/*                         Time step                         */
/*-----------------------------------------------------------*/
void simulation::tstep()
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

	if ((outside_world.rows() > 0) && (travelling_pop.rows() > 0)) {
		set_destination(population, destinations);
		check_at_destination(population, destinations, Config.wander_factor_dest);
	}

	if ((outside_world.rows() > 0) && (at_destination.rows() > 0)) {
		//keep them at destination
		keep_at_destination(population, Config.isolation_bounds);
	}
	//======================================================================================//
	//gravity wells
	if (Config.gravity_strength > 0.001) {
		tie(population, last_step_change) = update_gravity_forces(population,
			time, last_step_change, &my_rand, Config.wander_step_size,
			Config.gravity_strength, Config.wander_step_duration);
	}
	//======================================================================================//
	//activate social distancing above a certain infection threshold

	if (!(above_act_thresh) && (Config.social_distance_threshold_on > 0)) {
		// If not previously above infection threshold activate when threshold reached
		if (Config.thresh_type == "hospitalized") {
			above_act_thresh = population(select_rows(population.col(11) == 1), { 11 }).sum() >= Config.social_distance_threshold_on;
		}
		else if (Config.thresh_type == "infected") {
			above_act_thresh = population(select_rows(population.col(6) == 1), { 11 }).sum() >= Config.social_distance_threshold_on;
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
		above_deact_thresh = population(select_rows(cond), Eigen::all).count() <= Config.social_distance_threshold_off;
	}

	bool act_social_distancing;

	// activate social distancing at the onset of infection
	if (!Config.SD_act_onset) {
		act_social_distancing = ((above_act_thresh) && !(above_deact_thresh) && (pop_infected.count() > 0));
	}
	// activate social distancing from start of simulation
	else if (Config.SD_act_onset) {
		act_social_distancing = ((above_act_thresh) && !(above_deact_thresh));
	}

	//activate social distancing only for compliant individuals
	if ((Config.social_distance_factor > 0) && (act_social_distancing)) {

		ArrayXXb cond(Config.pop_size, 2);
		cond << (population.col(17) == 0), (population.col(11) == 0);
		population(select_rows(cond), Eigen::all) = update_repulsive_forces(population(select_rows(cond), Eigen::all), 
																			Config.social_distance_factor);

		if (population.col(15).isNaN().any()) {
			cout << "Infinite repulsive forces!" << endl;
			throw "Infinite repulsive forces!";
		}
	
	}

	//======================================================================================//
	//out of bounds
	//define bounds arrays, excluding those who are marked as having a custom destination
	if (inside_world.rows() > 0) {
		Eigen::ArrayXXf _xbounds(inside_world.rows(), 2), _ybounds(inside_world.rows(), 2);
		double buffer = 0.0;

		_xbounds << Eigen::ArrayXf::Ones(inside_world.rows(), 1) * (Config.xbounds[0] + buffer),
					Eigen::ArrayXf::Ones(inside_world.rows(), 1) * (Config.xbounds[1] - buffer);

		_ybounds << Eigen::ArrayXf::Ones(inside_world.rows(), 1) * (Config.ybounds[0] + buffer),
					Eigen::ArrayXf::Ones(inside_world.rows(), 1) * (Config.ybounds[1] - buffer);

		population(select_rows(population.col(11) == 0), Eigen::all) = update_wall_forces(population(select_rows(population.col(11) == 0), Eigen::all),
																				  _xbounds, _ybounds);

		if (population.col(15).isNaN().any()) {
			cout << "Infinite wall forces!" << endl;
			throw "Infinite wall forces!";
		}
	}

	if (population.col(1).isNaN().any()) {
		cout << "================================" << endl;
		cout << "Saving random seed state" << endl;
		my_rand.save_state();
		cout << "random_check: " << my_rand.rand() << endl;
		cout << "Division by zero condition!" << endl;
		throw "Division by zero condition!";
	}

	//======================================================================================//
	//update velocities
	ArrayXXb cond(Config.pop_size, 2);
	cond << (population.col(11) == 0), (population.col(12) == 1);

	population(select_rows_any(cond), Eigen::all) = update_velocities(population(select_rows_any(cond), Eigen::all),
																	  Config.max_speed, Config.dt);

	//for dead ones : set velocity and social distancing to 0 for dead ones
	population(select_rows(population.col(6) == 3), { 3,4 }) = 0;
	population(select_rows(population.col(6) == 3), { 17 }) = 1;

	//update positions
	population = update_positions(population, Config.dt);

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
	infect(population, destinations, Config, frame, &my_rand, Config.self_isolate,
			Config.isolation_bounds, 1, Config.self_isolate_proportion, act_testing);

	// recover and die
	recover_or_die(population, frame, Config, &my_rand);

	//======================================================================================//
	//send cured back to population if self isolation active
	//perhaps put in recover or die class
	//send cured back to population
	population(select_rows(population.col(6) == 2), { 11 }) = 0;

	//======================================================================================//
	//update population statistics
	pop_tracker.update_counts(population, frame);

	//======================================================================================//
	//visualise
	if ((Config.visualise) && ((frame % Config.visualise_every_n_frame) == 0)) {
		if (Config.n_plots == 1) {
			vis.draw_tstep_scatter(Config, population, pop_tracker, frame);
		}
		else if (Config.n_plots == 2) {
			vis.draw_tstep(Config, population, pop_tracker, frame);
		}
	}

	//report stuff to console
	if ((Config.verbose) && ((frame % Config.report_freq) == 0)) {
		cout << frame;
		cout << ": S: " << pop_tracker.susceptible.back();
		cout << ": I: " << pop_tracker.infectious.back();
		cout << ": R: " << pop_tracker.recovered.back();
		cout << ": in treatment: " << pop_hospitalized.rows();
		cout << ": F: " << pop_tracker.fatalities.back();
		cout << ": of total: " << Config.pop_size;
		if (Config.track_position) {
			cout << ": D: " << pop_tracker.distance_travelled.back()*100;
		}
		if (Config.track_GC) {
			cout << ": GC: " << pop_tracker.mean_perentage_covered.back()*100;
		}
		cout << " time: " << tc.toc() << " ms";
		cout << endl;
	}


	//save popdata if required
	if ((Config.save_pop) && ((frame % Config.save_pop_freq) == 0)) {
		save_population(population, frame, Config.save_pop_folder);
	}
	//run callback
	callback();

	//======================================================================================//
	//update frame
	frame += 1;
	time += Config.dt;
}

/*-----------------------------------------------------------*/
/*                   Placeholder function                    */
/*-----------------------------------------------------------*/
void simulation::callback()
{
	/*placeholder function that can be overwritten.

	By overwriting this method any custom behavior can be implemented.
	The method is called after every simulation timestep.
	*/

	if (frame == 50) {
		cout << "infecting person (Patient Zero)" << endl;

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
			ArrayXXf::Index minRow;
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
void simulation::run()
{
	/* run simulation */

	if (Config.visualise) {

		if (Config.n_plots == 1) {
			vis.build_fig_scatter(Config);
		}
		else if (Config.n_plots == 2) {
			vis.build_fig(Config);
		}

	}

	int i = 0;

	ArrayXXb cond(Config.pop_size, 2);

	while (i < Config.simulation_steps) {

		tstep();

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

	if (Config.plot_last_tstep) {
		vis.build_fig_SIR(Config);
		vis.draw_SIRonly(Config, population, pop_tracker, (frame - 1) );
	}

	if (Config.save_data) {
		save_data(population, pop_tracker);
	}

	// report outcomes
	if (Config.verbose) {

		cond << (population.col(6) == 1), (population.col(6) == 4);

		Eigen::ArrayXXf pop_susceptible = population(select_rows(population.col(6) == 0), Eigen::all); // healthy individuals
		Eigen::ArrayXXf pop_recovered = population(select_rows(population.col(6) == 2), Eigen::all); // recovered individuals
		Eigen::ArrayXXf pop_fatality = population(select_rows(population.col(6) == 3), Eigen::all); // dead individuals
		Eigen::ArrayXXf pop_asymptomatic = population(select_rows(population.col(6) == 4), Eigen::all); // asymptomatic individuals
		Eigen::ArrayXXf pop_infectious = population(select_rows_any(cond), Eigen::all); // infectious individuals


		cout << "\n\n" << "-----stopping-----" << endl;
		cout << "total timesteps taken: " << to_string(frame) << endl;
		cout << "total fatalities: " << to_string(pop_fatality.rows()) << endl;
		cout << "total recovered: " << to_string(pop_recovered.rows()) << endl;
		cout << "total infected: " << to_string(pop_infected.rows()) << endl;
		cout << "total infectious: " << to_string(pop_infectious.rows()) << endl;
		cout << "total unaffected: " << to_string(pop_susceptible.rows()) << endl;
		if (Config.track_GC) {
			cout << "Mean % explored: " << pop_tracker.mean_perentage_covered.back() << endl;
		}
	}

}

/*-----------------------------------------------------------*/
/*                        Destructor                         */
/*-----------------------------------------------------------*/
simulation::~simulation()
{
}

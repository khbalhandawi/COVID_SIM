// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Configuration.h"
#include "Population_trackers.h"
#include "utilities.h"
#include "motion.h"
#include "RandomDevice.h"
#include "visualizer.h"
#include "simulation.h"

#include <chrono>
#include <windows.h>
#include <iomanip>

using namespace std;

int main()
{
	std::cout << "Hello World!\n";

	// initialize
	Configuration Config;

	// set number of simulation steps
	Config.simulation_steps = 3000;
	Config.pop_size = 1000;
	Config.n_gridpoints = 10;
	Config.track_position = false;
	Config.track_GC = true;
	Config.update_every_n_frame = 1;
	Config.endif_no_infections = false;
	Config.SD_act_onset = true;
	Config.patient_Z_loc = "central";

	double area_scaling = 1.0 / double(Config.pop_size) / 600.0;
	double distance_scaling = 1.0 / sqrt(double(Config.pop_size) / 600.0);
	double force_scaling = pow(distance_scaling,4);
	double count_scaling = double(Config.pop_size) / 600.0;

	// set visuals
	Config.plot_style = "default"; // can also be dark
	Config.plot_text_style = "default"; // can also be LaTeX
	Config.visualise = false;
	Config.add_cross = true; // plot a cross
	Config.visualise_every_n_frame = 1;
	Config.n_plots = 1; // only scatter plot
	Config.plot_last_tstep = true;
	Config.verbose = true;
	Config.report_freq = 50; // report results every 50 frames
	Config.save_plot = false;
	Config.save_data = false;
	// Config.marker_size = (2700 - Config.pop_size) / 140;
	Config.marker_size = 5;

	// set infection parameters
	Config.infection_chance = 0.3;
	Config.infection_range = 0.03 * distance_scaling;
	Config.mortality_chance = 0.09; // global baseline chance of dying from the disease
	Config.incubation_period = 5;

	// set movement parameters
	Config.speed = 0.15 * distance_scaling;
	Config.max_speed = 0.3 * distance_scaling;
	Config.dt = 0.01;

	Config.wander_step_size = 0.01 * distance_scaling;
	Config.gravity_strength = 0;
	Config.wander_step_duration = Config.dt * 10;

	// run 0 (Business as usual)
	//Config.social_distance_factor = 0.0001 * 0.0 * force_scaling;

	// run 1 (social distancing)
	//Config.social_distance_factor = 0.0001 * 0.3 * force_scaling;

	// run 2 (social distancing with violators)
	//Config.social_distance_factor = 0.0001 * 0.3 * force_scaling;
	//Config.social_distance_violation = 20;

	// run 3 (self-isolation scenario)
	Config.healthcare_capacity = 150;
	Config.wander_factor_dest = 0.1;
	Config.set_self_isolation(100, 1.0, { -0.26, 0.02, 0.0, 0.28 }, false);

	// run 4 (self - isolation scenario with social distancing)
	Config.social_distance_factor = 0.0001 * 0.1 * force_scaling;
	Config.social_distance_threshold_on = 15; // number of people
	Config.testing_threshold_on = 15; // number of people

	simulation sim(Config);
	//sim.population_init();
	//sim.initialize_simulation();
	// run, hold CTRL + C in terminal to end scenario early
	sim.run();
}

//int main()
//{
//	std::cout << "hello world!\n";
//
//	unsigned long seed = static_cast<uint32_t>(high_resolution_clock::now().time_since_epoch().count());
//
//	RandomDevice my_rand(seed);
//
//	int n_choices = 5;
//
//	Eigen::ArrayXf input(7);
//	input << 2,3,4,5,6,9,10 ;
//
//	vector<int> indices = sequence(0, input.rows(), 1);
//	shuffle(indices.begin(), indices.end(), my_rand.engine);
//
//	cout << "======================" << endl;
//	for (int i = 0; i < indices.size(); i++) {
//		cout << indices[i] << endl;
//	}
//	cout << "-------------" << endl;
//
//	indices = slice(indices, 0, n_choices);
//
//	cout << "======================" << endl;
//	for (int i = 0; i < indices.size(); i++) {
//		cout << indices[i] << endl;
//	}
//	cout << "-------------" << endl;
//
//	Eigen::ArrayXf output_f = input(indices);
//	Eigen::VectorXi output = output_f.col(0).cast<int>();
//
//	cout << output << endl;
//
//	Eigen::ArrayXf duplicate(10);
//
//	duplicate << 10, 10, 20, 20, 30, 30, 30, 10, 20, 40;
//
//	vector<float> duplicate_vec(duplicate.rows()); Map<ArrayXf>(&duplicate_vec[0], duplicate.rows(), 1) = duplicate;
//
//	unique_elements(duplicate_vec);
//
//	for (int i : duplicate_vec)
//		std::cout << i << " ";
//	std::cout << "\n";
//
//	Configuration Config;
//	double mortality_chance = Config.mortality_chance;
//
//	compute_mortality(56, mortality_chance, Config.risk_age, Config.critical_age,
//		Config.critical_mortality_chance, Config.risk_increase);
//
//	cout << "mortality: " << mortality_chance << endl;
//
//	Eigen::ArrayXf p1(2), p2(2);
//	p1 << 0.74898833, 0.77640605;
//	p2 << 0.748951077, 0.776557982;
//
//	Eigen::ArrayXXf test(2,2);
//	int id;
//
//	test << p1, p2;
//
//	cout << test << endl;
//
//	cout << p1 - p2 << endl;
//	cout << pairwise_dist(test, id) << endl;
//
//
//}

	//Configuration config;
	//config.simulation_steps = 2000;
	//config.pop_size = 1000;
	//config.n_gridpoints = 33;
	//config.xbounds = { 0.0,1.0 };
	//config.ybounds = { 0.0,1.0 };
	//config.social_distance_violation = 1000;
	//config.track_GC = false;
	//config.update_every_n_frame = 1;
	//config.max_speed = 0.25;
	//config.dt = 0.01;
	//config.social_distance_factor = 0.3;

	///*-----------------------------------------------------------*/
	///*                  Debug individual methods                 */
	///*-----------------------------------------------------------*/
	//// seed random generator
	///* using nano-seconds instead of seconds */
	//unsigned long seed = static_cast<uint32_t>(high_resolution_clock::now().time_since_epoch().count());
	//RandomDevice my_rand(seed);

	//// debug color palette
	//vector<string> x = config.get_palette();

	//// debug lockdown
	//config.set_lockdown(&my_rand, 0.9, 0.5);
	////config.set_lockdown();

	//// debug population initialization
	//Eigen::ArrayXXf population = initialize_population(config, &my_rand);

	//// debug ground covered initialization
	//Eigen::ArrayXXf grid_coords, ground_covered;
	//tie(grid_coords, ground_covered) = initialize_ground_covered_matrix(config.pop_size, config.n_gridpoints, config.xbounds, config.ybounds);

	//Population_trackers pop_tracker(config, grid_coords, ground_covered);
	//pop_tracker.update_counts(population, 0);

	//save_data(population, pop_tracker);
	//save_population(population, 1, "population");

	//// test RNG
	////for (int i = 0; i < 100; i++) { 
	////	cout << my_rand.rand() << endl;
	////}

	//Eigen::ArrayXXf _xbounds(config.pop_size, 2), _ybounds(config.pop_size, 2);
	//double buffer = 0.0;

	//_xbounds << Eigen::ArrayXf::Ones(config.pop_size, 1) * (config.xbounds[0] + buffer),
	//			Eigen::ArrayXf::Ones(config.pop_size, 1) * (config.xbounds[1] - buffer);
	//
	//_ybounds << Eigen::ArrayXf::Ones(config.pop_size, 1) * (config.ybounds[0] + buffer),
	//			Eigen::ArrayXf::Ones(config.pop_size, 1) * (config.ybounds[1] - buffer);
	//
	//population = update_wall_forces(population, _xbounds, _ybounds);
	//population = update_velocities(population, config.max_speed, config.dt);
	//population = update_repulsive_forces(population, config.social_distance_factor);

	//double time = 1.0, last_step_change = 0.1;
	//tie(population, last_step_change) = update_gravity_forces(population, time, last_step_change, &my_rand);

	//vector<double> outputs = get_motion_parameters(config.xbounds[0], config.ybounds[0], config.xbounds[1], config.ybounds[1]);

	///*-----------------------------------------------------------*/
	///*                 Minimum working simulation                */
	///*-----------------------------------------------------------*/
	//// seed random generator
	///* using nano-seconds instead of seconds */
	//unsigned long seed = static_cast<uint32_t>(high_resolution_clock::now().time_since_epoch().count());
	//RandomDevice my_rand(seed);

	//Eigen::ArrayXXf population = initialize_population(config, &my_rand);
	//population(Eigen::all, { 1,2 }) << 0.0849588, 0.213444,
	//0.437875, 0.59214,
	//0.73224, 0.45096,
	//0.674879, 0.786657,
	//0.903503, 0.497677,
	//0.470838, 0.872818,
	//0.205956, 0.306843,
	//0.219279, 0.726152,
	//0.230023, 0.39993,
	//0.601939, 0.134491;

	//cout << population(Eigen::all, { 1,2 }) << endl;

	//Eigen::ArrayXXf grid_coords, ground_covered;
	//tie(grid_coords, ground_covered) = initialize_ground_covered_matrix(config.pop_size, config.n_gridpoints, config.xbounds, config.ybounds);
	//Population_trackers pop_tracker(config, grid_coords, ground_covered);

	//Eigen::ArrayXXf _xbounds(config.pop_size, 2), _ybounds(config.pop_size, 2);
	//double buffer = 0.0;

	//_xbounds << Eigen::ArrayXf::Ones(config.pop_size, 1) * (config.xbounds[0] + buffer),
	//Eigen::ArrayXf::Ones(config.pop_size, 1) * (config.xbounds[1] - buffer);

	//_ybounds << Eigen::ArrayXf::Ones(config.pop_size, 1) * (config.ybounds[0] + buffer),
	//Eigen::ArrayXf::Ones(config.pop_size, 1) * (config.ybounds[1] - buffer);

	//for (int i = 0; i < config.simulation_steps; i++)
	//{
	//	population = update_wall_forces(population, _xbounds, _ybounds);
	//	population = update_repulsive_forces(population, config.social_distance_factor);

	//	population = update_velocities(population, config.max_speed, config.dt);
	//	population = update_positions(population, config.dt);

	//	pop_tracker.update_counts(population, i);

	//	if ((i % 50) == 0) { cout << i << endl; }
	//}

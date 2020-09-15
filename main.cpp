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
#include <sstream>
#include <iterator>
#include <iostream>
#include <fstream>

using namespace std;

/*-----------------------------------------------------------*/
/*              Post process simulation results              */
/*-----------------------------------------------------------*/
vector<double> processInput(int i,simulation *sim, ofstream *file)
{
	sim->run();
	
	int infected, fatalities, SD_thresh, E, T, f;
	double mean_distance, mean_GC, SD;

	SD = sim->Config.social_distance_factor / 0.0001;
	SD_thresh = sim->Config.social_distance_threshold_on;
	E = sim->Config.social_distance_violation;
	T = sim->Config.number_of_tests;

	infected = *max_element(sim->pop_tracker.infectious.begin(), sim->pop_tracker.infectious.end());
	fatalities = sim->pop_tracker.fatalities.back();
	mean_distance = (sim->pop_tracker.distance_travelled.back() / double(sim->frame)) * 100.0 * 2000;
	mean_GC = (sim->pop_tracker.mean_perentage_covered.back() / double(sim->frame)) * 100.0 * 2000;

	f = sim->frame;

	int n_vars = 4;
	vector<double> matrix_out, matrix_opt; // unstripped matrix
	matrix_out.push_back(i);
	matrix_out.push_back(SD);
	matrix_out.push_back(SD_thresh);
	matrix_out.push_back(E);
	matrix_out.push_back(T);
	matrix_out.push_back(infected); matrix_opt.push_back(infected);
	matrix_out.push_back(fatalities); matrix_opt.push_back(fatalities);
	matrix_out.push_back(mean_distance); matrix_opt.push_back(mean_distance);
	matrix_out.push_back(mean_GC); matrix_opt.push_back(mean_GC);
	matrix_out.push_back(f);

	// Convert int to ostring stream
	ostringstream oss;
	oss.precision(11);
	if (!matrix_out.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		copy(matrix_out.begin(), matrix_out.end() - 1,
			ostream_iterator<double>(oss, ","));

		// Now add the last element with no delimiter
		oss << matrix_out.back();
	}

	file->precision(11);
	// Write ostring stream to file
	if (file->is_open())
	{
		(*file) << oss.str() << '\n';
	}
	
	return matrix_opt;

}

/*-----------------------------------------------------------*/
/*                    Load configuration                     */
/*-----------------------------------------------------------*/
void load_config(Configuration *config, const char *config_file)
{	
	map<string,string Configuration::*> mapper;
	mapper["simulation_steps"] = &Configuration::simulation_steps_in;
	mapper["pop_size"] = &Configuration::pop_size_in;
	mapper["n_gridpoints"] = &Configuration::n_gridpoints_in;
	mapper["track_position"] = &Configuration::track_position_in;
	mapper["track_GC"] = &Configuration::track_GC_in;
	mapper["update_every_n_frame"] = &Configuration::update_every_n_frame_in;
	mapper["endif_no_infections"] = &Configuration::endif_no_infections_in;
	mapper["SD_act_onset"] = &Configuration::SD_act_onset_in;
	mapper["patient_Z_loc"] = &Configuration::patient_Z_loc_in;
	mapper["plot_style"] = &Configuration::plot_style_in;
	mapper["plot_text_style"] = &Configuration::plot_text_style_in;
	mapper["visualise"] = &Configuration::visualise_in;
	mapper["add_cross"] = &Configuration::add_cross_in;
	mapper["visualise_every_n_frame"] = &Configuration::visualise_every_n_frame_in;
	mapper["n_plots"] = &Configuration::n_plots_in;
	mapper["plot_last_tstep"] = &Configuration::plot_last_tstep_in;
	mapper["verbose"] = &Configuration::verbose_in;
	mapper["report_freq"] = &Configuration::report_freq_in;
	mapper["save_plot"] = &Configuration::save_plot_in;
	mapper["save_data"] = &Configuration::save_data_in;
	mapper["marker_size"] = &Configuration::marker_size_in;
	mapper["infection_chance"] = &Configuration::infection_chance_in;
	mapper["infection_range"] = &Configuration::infection_range_in;
	mapper["mortality_chance"] = &Configuration::mortality_chance_in;
	mapper["incubation_period"] = &Configuration::incubation_period_in;
	mapper["speed"] = &Configuration::speed_in;
	mapper["max_speed"] = &Configuration::max_speed_in;
	mapper["dt"] = &Configuration::dt_in;
	mapper["wander_step_size"] = &Configuration::wander_step_size_in;
	mapper["gravity_strength"] = &Configuration::gravity_strength_in;
	mapper["mortality_chance"] = &Configuration::mortality_chance_in;
	mapper["mortality_chance"] = &Configuration::mortality_chance_in;
	mapper["wander_step_duration"] = &Configuration::wander_step_duration_in;
	mapper["constant_seed"] = &Configuration::constant_seed_in;
	mapper["thresh_type"] = &Configuration::thresh_type_in;
	mapper["social_distance_threshold_off"] = &Configuration::social_distance_threshold_off_in;
	mapper["social_distance_threshold_on"] = &Configuration::social_distance_threshold_on_in;
	mapper["trace_path"] = &Configuration::trace_path_in;

	ifstream file(config_file); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
	string line, value, value_str;

	while (file.good()) {
		getline(file, line, '\n');
		istringstream is_line(line);
		vector<string> key_value_pair;

		while (getline(is_line, value, '=')) {
			key_value_pair.push_back(value.c_str());
		}
		config->*(mapper[key_value_pair[0]])=key_value_pair[1];
	}
}

/*-----------------------------------------------------------*/
/*                    Main execution call                    */
/*-----------------------------------------------------------*/
int main(int argc, char* argv[])
{
	
	bool debug;
	double SD;
	int run, n_violators, test_capacity, healthcare_capacity;

	// Check if this is an external system call
    if (argc > 2) {
		debug = false;
	} 
	else {
		debug = true;
	}

	if (debug) {
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
		double force_scaling = pow(distance_scaling,2);
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

		// seed random generator
		/* using nano-seconds instead of seconds */
		unsigned long seed = static_cast<uint32_t>(high_resolution_clock::now().time_since_epoch().count());

		simulation sim(Config, seed);
		//sim.population_init();
		//sim.initialize_simulation();
		// run, hold CTRL + C in terminal to end scenario early
		sim.run();
	} 
	else if (!debug) {

		/*-----------------------------------------------------------*/
		/*                Read command line arguments                */
		/*-----------------------------------------------------------*/

		run = stoi(argv[1]);

		// Model variables
		n_violators = stoi(argv[2]);
		SD = atof(argv[3]);
		test_capacity = stoi(argv[4]);

		// Model parameters
		healthcare_capacity = stoi(argv[5]);

		// Log file name
		string log_file;
		if (argc == 7) {
			string filename = argv[6];
			log_file = "data/" + filename;
		} 
		else { // if no file name argument provided use defaults
			log_file = "data/opt_run.log";
		}

		// Display input arguments
		cout << "\n" << "================= starting =================" << endl;
		cout << "E: " << n_violators << " | SD: " << SD << " | T: " << test_capacity << " | H_c: " << healthcare_capacity << " | output: " << log_file <<"\n";

		/*-----------------------------------------------------------*/
		/*            Simulation configuration variables             */
		/*-----------------------------------------------------------*/
		// initialize
		Configuration Config;

		const char config_file[] = "Configuration.ini";

		load_config(&Config, config_file);
		Config.set_from_file();

		double area_scaling = 1.0 / double(Config.pop_size) / 600.0;
		double distance_scaling = 1.0 / sqrt(double(Config.pop_size) / 600.0);
		double force_scaling = pow(distance_scaling,2);
		double count_scaling = double(Config.pop_size) / 600.0;

		/*-----------------------------------------------------------*/
		/*                      Design variables                     */
		/*-----------------------------------------------------------*/

        Config.social_distance_factor = 1e-6 * SD * force_scaling;
        Config.social_distance_violation = n_violators; // number of people
        Config.healthcare_capacity = healthcare_capacity;

		if (test_capacity > 0) {
			Config.testing_threshold_on = 15; // number of people 
			Config.wander_factor_dest = 0.1;
			Config.set_self_isolation(test_capacity, 1.0, { -0.26, 0.02, 0.0, 0.28 }, false);
		}

		/*-----------------------------------------------------------*/
		/*                    Log blackbox outputs                   */
		/*-----------------------------------------------------------*/

		// seed random generator
		/* using nano-seconds instead of seconds */
		// unsigned long seed = static_cast<uint32_t>(high_resolution_clock::now().time_since_epoch().count());
		unsigned long seed = run;
		simulation sim(Config, seed);

		check_folder("data");
		string filename = "matlab_out_Blackbox.log";
		string full_filename = "data/" + filename;

		// Output evaluation to file

		vector<double> matrix_opt;
		ofstream output_file;

		if (run == 0) {
			output_file.open(log_file, ofstream::out);

			output_file.precision(11);
			output_file << "index,SD_factor,threshold,essential_workers,testing_capacity," << 
						   "n_infected,n_fatalaties,mean_distance,mean_GC,n_steps" << endl;
		}
		else {
			output_file.open(log_file, ofstream::app);
			output_file.precision(11);
		}

		matrix_opt = processInput(run, &sim, &output_file);
		output_file.close();

		double infected = matrix_opt[0]; 
		double fatalities = matrix_opt[1];
		double mean_distance = matrix_opt[2];
		double mean_GC = matrix_opt[3];
        
		double obj_1 = -mean_GC;
		double obj_2 = fatalities;
		double c1 = infected - healthcare_capacity;

		ofstream output_file_opt(full_filename);
		output_file_opt.precision(10); // number of decimal places to output
		output_file_opt << obj_1 << " " << obj_2 << " " << c1 << endl;
		output_file_opt.close();
		cout << "obj_1: " << obj_1 << " obj_2: " << obj_2 << " c1: " << c1 << endl;

	}

}

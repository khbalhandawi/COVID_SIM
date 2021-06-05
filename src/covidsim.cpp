// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Configuration.h"
#include "Population_trackers.h"
#include "utilities.h"
#include "motion.h"
#include "RandomDevice.h"
#include "simulation.h"

#include <iostream>
#include <fstream>
#include <map>

#ifdef __llvm__
#include <sys/stat.h>
#include <sys/types.h>
#include <mach-o/dyld.h>
#endif

using namespace std;

/*-----------------------------------------------------------*/
/*              Post process simulation results              */
/*-----------------------------------------------------------*/
vector<double> processInput(int i, COVID_SIM::simulation *sim, ofstream *file)
{
	sim->run();
	
	int infected, fatalities, SD_thresh, E, T, f;
	double mean_distance, mean_GC, SD;

	double distance_scaling = sim->Config.distance_scaling;
	double force_scaling = sim->Config.force_scaling;

	SD = sim->Config.social_distance_factor / (1e-6 * force_scaling);
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
void load_config(COVID_SIM::Configuration *config, const char *config_file)
{	
	std::map<string,string COVID_SIM::Configuration::*> mapper;
	mapper["simulation_steps"] = &COVID_SIM::Configuration::simulation_steps_in;
	mapper["pop_size"] = &COVID_SIM::Configuration::pop_size_in;
	mapper["n_gridpoints"] = &COVID_SIM::Configuration::n_gridpoints_in;
	mapper["track_position"] = &COVID_SIM::Configuration::track_position_in;
	mapper["track_GC"] = &COVID_SIM::Configuration::track_GC_in;
	mapper["update_every_n_frame"] = &COVID_SIM::Configuration::update_every_n_frame_in;
	mapper["endif_no_infections"] = &COVID_SIM::Configuration::endif_no_infections_in;
	mapper["SD_act_onset"] = &COVID_SIM::Configuration::SD_act_onset_in;
	mapper["patient_Z_loc"] = &COVID_SIM::Configuration::patient_Z_loc_in;
	mapper["patient_Z_time"] = &COVID_SIM::Configuration::patient_Z_time_in;
	mapper["plot_style"] = &COVID_SIM::Configuration::plot_style_in;
	mapper["plot_text_style"] = &COVID_SIM::Configuration::plot_text_style_in;
	mapper["visualise"] = &COVID_SIM::Configuration::visualise_in;
	mapper["add_cross"] = &COVID_SIM::Configuration::add_cross_in;
	mapper["visualise_every_n_frame"] = &COVID_SIM::Configuration::visualise_every_n_frame_in;
	mapper["n_plots"] = &COVID_SIM::Configuration::n_plots_in;
	mapper["plot_last_tstep"] = &COVID_SIM::Configuration::plot_last_tstep_in;
	mapper["verbose"] = &COVID_SIM::Configuration::verbose_in;
	mapper["report_freq"] = &COVID_SIM::Configuration::report_freq_in;
	mapper["save_plot"] = &COVID_SIM::Configuration::save_plot_in;
	mapper["save_data"] = &COVID_SIM::Configuration::save_data_in;
	mapper["marker_size"] = &COVID_SIM::Configuration::marker_size_in;
	mapper["infection_chance"] = &COVID_SIM::Configuration::infection_chance_in;
	mapper["infection_range"] = &COVID_SIM::Configuration::infection_range_in;
	mapper["mortality_chance"] = &COVID_SIM::Configuration::mortality_chance_in;
	mapper["incubation_period"] = &COVID_SIM::Configuration::incubation_period_in;
	mapper["speed"] = &COVID_SIM::Configuration::speed_in;
	mapper["max_speed"] = &COVID_SIM::Configuration::max_speed_in;
	mapper["dt"] = &COVID_SIM::Configuration::dt_in;
	mapper["wander_step_size"] = &COVID_SIM::Configuration::wander_step_size_in;
	mapper["gravity_strength"] = &COVID_SIM::Configuration::gravity_strength_in;
	mapper["mortality_chance"] = &COVID_SIM::Configuration::mortality_chance_in;
	mapper["mortality_chance"] = &COVID_SIM::Configuration::mortality_chance_in;
	mapper["wander_step_duration"] = &COVID_SIM::Configuration::wander_step_duration_in;
	mapper["constant_seed"] = &COVID_SIM::Configuration::constant_seed_in;
	mapper["thresh_type"] = &COVID_SIM::Configuration::thresh_type_in;
	mapper["social_distance_threshold_off"] = &COVID_SIM::Configuration::social_distance_threshold_off_in;
	mapper["social_distance_threshold_on"] = &COVID_SIM::Configuration::social_distance_threshold_on_in;
	mapper["trace_path"] = &COVID_SIM::Configuration::trace_path_in;
	mapper["write_bb_output"] = &COVID_SIM::Configuration::write_bb_output_in;
	mapper["social_distance_factor"] = &COVID_SIM::Configuration::social_distance_factor_in;
	mapper["social_distance_violation"] = &COVID_SIM::Configuration::social_distance_violation_in;
	mapper["healthcare_capacity"] = &COVID_SIM::Configuration::healthcare_capacity_in;
	mapper["number_of_tests"] = &COVID_SIM::Configuration::number_of_tests_in;
	mapper["self_isolate"] = &COVID_SIM::Configuration::self_isolate_in;
	mapper["wander_factor_dest"] = &COVID_SIM::Configuration::wander_factor_dest_in;
	mapper["isolation_bounds"] = &COVID_SIM::Configuration::isolation_bounds_in;
	mapper["self_isolate_proportion"] = &COVID_SIM::Configuration::self_isolate_proportion_in;
	mapper["xbounds"] = &COVID_SIM::Configuration::xbounds_in;
	mapper["ybounds"] = &COVID_SIM::Configuration::ybounds_in;
	mapper["x_plot"] = &COVID_SIM::Configuration::x_plot_in;
	mapper["y_plot"] = &COVID_SIM::Configuration::y_plot_in;
	mapper["traveling_infects"] = &COVID_SIM::Configuration::traveling_infects_in;
	mapper["save_pop"] = &COVID_SIM::Configuration::save_pop_in;
	mapper["save_ground_covered"] = &COVID_SIM::Configuration::save_ground_covered_in;
	mapper["save_pop_freq"] = &COVID_SIM::Configuration::save_pop_freq_in;
	mapper["infection_shape"] = &COVID_SIM::Configuration::infection_shape_in;
	mapper["track_R0"] = &COVID_SIM::Configuration::track_R0_in;
	mapper["update_R0_every_n_frame"] = &COVID_SIM::Configuration::update_R0_every_n_frame_in;
	mapper["testing_threshold_on"] = &COVID_SIM::Configuration::testing_threshold_on_in;
	mapper["platform"] = &COVID_SIM::Configuration::platform_in;
	mapper["wall_buffer"] = &COVID_SIM::Configuration::wall_buffer_in;
	mapper["bounce_buffer"] = &COVID_SIM::Configuration::bounce_buffer_in;

#ifdef __llvm__
    // get current working directory of executable
    char buf [PATH_MAX];
    std::uint32_t bufsize = PATH_MAX;
    if(!_NSGetExecutablePath(buf, &bufsize)) std::puts(buf);

    // Remove executable name from directory
    std::string buf_dir = buf;
    buf_dir = buf_dir.substr(0, buf_dir.find_last_of("\\/"));

    char final [256];
    std::sprintf (final, "%s/%s",buf_dir.c_str(),config_file);
    // std::cout << final << std::endl;
    
	ifstream file(final); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
#else
	ifstream file(config_file); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
#endif
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

#ifdef GPU_ACC
	cublasHandle_t handle;
	cublascheck(cublasCreate(&handle)); // construct cublas handle
#endif
	COVID_SIM::simulation* sim;

	if (debug) {
		
		// Display input arguments
		cout << "\n" << "/*---------------------------------------------------------------------------------*/";
		cout << "\n" << "/*  COVID SIM - Agent-based model of a pandemic in a population -                  */";
		cout << "\n" << "/*                                                                                 */";
		cout << "\n" << "/*  COVID SIM - version 1.0.0 has been created by                                  */";
		cout << "\n" << "/*                 Khalil Al Handawi           - McGill University                 */";
		cout << "\n" << "/*                                                                                 */";
		cout << "\n" << "/*  The copyright of COVID SIM - version 1.0.0 is owned by                         */";
		cout << "\n" << "/*                 Khalil Al Handawi           - McGill University                 */";
		cout << "\n" << "/*                 Michael Kokkolaras          - McGill University                 */";
		cout << "\n" << "/*                                                                                 */";
		cout << "\n" << "/*---------------------------------------------------------------------------------*/" << endl;

		/*-----------------------------------------------------------*/
		/*            Simulation configuration variables             */
		/*-----------------------------------------------------------*/
		// initialize
		COVID_SIM::Configuration Config;
		const char config_file[] = "configuration_debug.ini";

		load_config(&Config, config_file);
		Config.set_from_file();
		/*-----------------------------------------------------------*/
		/*                        Run blackbox                       */
		/*-----------------------------------------------------------*/

		// seed random generator
		/* using nano-seconds instead of seconds */
		unsigned long seed = static_cast<uint32_t>(chrono::high_resolution_clock::now().time_since_epoch().count());
#ifdef GPU_ACC
		sim = new COVID_SIM::simulation(Config, seed, handle);
#else
		sim = new COVID_SIM::simulation(Config, seed);
#endif
		sim->run();

		delete sim;
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

		/*-----------------------------------------------------------*/
		/*            Simulation configuration variables             */
		/*-----------------------------------------------------------*/
		// initialize
		COVID_SIM::Configuration Config;

		const char config_file[] = "configuration.ini";

		load_config(&Config, config_file);
		Config.set_from_file();

		if (Config.verbose) {
			// Display input arguments
			cout << "\n" << "================= starting =================" << endl;
			cout << "E: " << n_violators << " | SD: " << SD << " | T: " << test_capacity << " | H_c: " << healthcare_capacity << " | output: " << log_file << "\n";
			cout << "Config loaded!" << endl;
		}

		/*-----------------------------------------------------------*/
		/*                      Design variables                     */
		/*-----------------------------------------------------------*/

        Config.social_distance_factor = 1e-6 * SD * Config.force_scaling;
        Config.social_distance_violation = n_violators; // number of people
        Config.healthcare_capacity = healthcare_capacity;
		Config.number_of_tests = test_capacity;

		/*-----------------------------------------------------------*/
		/*                    Log blackbox outputs                   */
		/*-----------------------------------------------------------*/

		// seed random generator
		/* using nano-seconds instead of seconds */
		unsigned long seed = static_cast<uint32_t>(chrono::high_resolution_clock::now().time_since_epoch().count());
#ifdef GPU_ACC
		sim = new COVID_SIM::simulation(Config, seed, handle);
#else
		sim = new COVID_SIM::simulation(Config, seed);
#endif
		if (Config.verbose) {
			cout << "initialized simulation" << endl;
		}
		COVID_SIM::check_folder("data");
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

		matrix_opt = processInput(run, sim, &output_file);
		output_file.close();

		double infected = matrix_opt[0]; 
		double fatalities = matrix_opt[1];
		double mean_distance = matrix_opt[2];
		double mean_GC = matrix_opt[3];
        
		double obj_1 = -mean_GC;
		double obj_2 = fatalities;
		double c1 = infected - healthcare_capacity;

		if (Config.write_bb_output) {
			ofstream output_file_opt(full_filename);
			output_file_opt.precision(10); // number of decimal places to output
			output_file_opt << obj_1 << " " << obj_2 << " " << c1 << endl;
			output_file_opt.close();
			if (Config.verbose) {
				cout << "obj_1: " << obj_1 << " obj_2: " << obj_2 << " c1: " << c1 << endl;
			}
		}

		delete sim;

	}

#ifdef GPU_ACC
	cublasDestroy(handle); // destroy cublas handle to avoid malloc errors
#endif

}

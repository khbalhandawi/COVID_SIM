/*---------------------------------------------------------------------------------*/
/*  COVID SIM - Agent-based model of a pandemic in a population -                  */
/*                                                                                 */
/*  COVID SIM - version 1.0.0 has been created by                                  */
/*                 Khalil Al Handawi           - McGill University                 */
/*                                                                                 */
/*  The copyright of NOMAD - version 3.9.1 is owned by                             */
/*                 Khalil Al Handawi           - McGill University                 */
/*                 Michael Kokkolaras          - McGill University                 */
/*                                                                                 */
/*                                                                                 */
/*  Contact information:                                                           */
/*    McGill University - Systems Optimization Lab (SOL)                           */
/*    Macdonald Engineering Building, 817 Sherbrooke Street West,                  */
/*    Montreal (Quebec) H3A 0C3 Canada                                             */
/*    e-mail: khalil.alhandawi@mail.mcgill.ca                                      */
/*    phone : 1-514-398-2343                                                       */
/*                                                                                 */
/*  This program is free software: you can redistribute it and/or modify it        */
/*  under the terms of the GNU Lesser General Public License as published by       */
/*  the Free Software Foundation, either version 3 of the License, or (at your     */
/*  option) any later version.                                                     */
/*                                                                                 */
/*  This program is distributed in the hope that it will be useful, but WITHOUT    */
/*  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or          */
/*  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License    */
/*  for more details.                                                              */
/*                                                                                 */
/*  You should have received a copy of the GNU Lesser General Public License       */
/*  along with this program. If not, see <http://www.gnu.org/licenses/>.           */
/*                                                                                 */
/*---------------------------------------------------------------------------------*/

/**
 \file   Configuration.cpp
 \brief  Simulation configuration parameters and modifiers (implementation)
 \author Khalil Al Handawi
 \date   2010-08-25
 \see    Configuration.h
 */

#include "Configuration.h"
#include "Population_trackers.h"

/*-----------------------------------------------------------*/
/*                       Constructor                         */
/*-----------------------------------------------------------*/
Configuration::Configuration()
{
	// simulation variables
	simulation_steps = 10000; simulation_steps_in = "10000"; // total simulation steps performed
	tstep = 0; tstep_in = "0"; // current simulation timestep
	save_data = false; save_data_in = "false"; // whether to dump data at end of simulation
	save_pop = false; save_pop_in = "false"; // whether to save population matrix every "save_pop_freq" timesteps
	save_ground_covered = false; save_ground_covered_in = "false"; // whether to save ground covered matrix every "save_pop_freq" timesteps
	save_pop_freq = 10; save_pop_freq_in = "10"; // population data will be saved every "n" timesteps.Default: 10
	save_pop_folder = "population"; save_pop_folder_in = "population"; // folder to write population timestep data to
	endif_no_infections = true; endif_no_infections_in = "true"; // whether to stop simulation if no infections remain
	write_bb_output = true; write_bb_output_in = "true"; // report results to black box output file for optimization
	// scenario flags
	traveling_infects = false; traveling_infects_in = "false";
	self_isolate = false; self_isolate_in = "false";
	lockdown = false; lockdown_in = "false";
	lockdown_compliance = 0.95; lockdown_compliance_in = "0.95"; // fraction of the population that will obey the lockdown

	// world variables, defines where population can and cannot roam
	xbounds = { 0.02, 0.98 }; xbounds_in = "0.02, 0.98";
	ybounds = { 0.02, 0.98 }; ybounds_in = "0.02, 0.98";
	n_gridpoints = 10; n_gridpoints_in = "10"; //  resolution of 2D grid for tracking population position
	track_position = true; track_position_in = "true";
	track_GC = false; track_GC_in = "false";
	track_R0 = false; track_R0_in = "false";
	update_every_n_frame = 1; update_every_n_frame_in = "1";
	update_R0_every_n_frame = 1; update_R0_every_n_frame_in = "1";

	// visualisation variables
	visualise = true; visualise_in = "true"; // whether to visualise the simulation
	platform = "Qt"; platform_in = "Qt"; // which platform to use for visualization
	add_cross = false; add_cross_in = "false"; // draws a red cross on hospital
	visualise_every_n_frame = 1; visualise_every_n_frame_in = "1"; // Frequency of plot update
	plot_mode = "sir"; plot_mode_in = "sir"; // default or sir
	n_plots = 2; n_plots_in = "2"; // number of subplots
	plot_last_tstep = true; plot_last_tstep_in = "true"; // plot last frame SIR
	trace_path = false; trace_path_in = "false"; // trace path of a single individual
	// size of the simulated world in coordinates
	x_plot = { 0, 1 }; x_plot_in = "0, 1";
	y_plot = { 0, 1 }; y_plot_in = "0, 1";
	save_plot = false; save_plot_in = "false";
	plot_path = "render"; plot_path_in = "render"; // folder where plots are saved to
	plot_style = "default"; plot_style_in = "default"; // can be default, dark, ...
	plot_text_style = "default"; plot_text_style_in = "default"; // can be default, LaTeX, ...
	colorblind_mode = false; colorblind_mode_in = "false";
	// if colorblind is enabled, set type of colorblindness
	// available: deuteranopia, protanopia, tritanopia.defauld = deuteranopia
	colorblind_type = "deuteranopia"; colorblind_type_in = "deuteranopia";
	verbose = true; verbose_in = "true"; // output stats to console
	report_freq = 1; report_freq_in = "1"; // report results every frame to console
	report_status = false; report_status_in = "false"; // output stats to console
	marker_size = 15; marker_size_in = "15"; // markersize for plotting individuals

	// population variables
	pop_size = 2000; pop_size_in = "2000";
	mean_age = 45; mean_age_in = "45";
	max_age = 105; max_age_in = "105";
	age_dependent_risk = true; age_dependent_risk_in = "true"; // whether risk increases with age
	risk_age = 55; risk_age_in = "55"; // age where mortality risk starts increasing
	critical_age = 75; critical_age_in = "75"; // age at and beyond which mortality risk reaches maximum
	critical_mortality_chance = 0.1; critical_mortality_chance_in = "0.1"; // maximum mortality risk for older age
	risk_increase = "quadratic"; risk_increase_in = "quadratic"; // whether risk between risk and critical age increases "linear" or "quadratic"

	// movement variables
	// the proportion of the population that practices social distancing, simulated
	// by them standing still
	proportion_distancing = 0; proportion_distancing_in = "0";
	social_distance_factor = 0.0; social_distance_factor_in = "0.0";
	speed = 0.01; speed_in = "0.01"; // average speed of population
	max_speed = 1.0; max_speed_in = "1.0"; // average speed of population
	dt = 0.01; dt_in = "0.01"; // average speed of population

	wander_step_size = 0.01; wander_step_size_in = "0.01";
	gravity_strength = 1; gravity_strength_in = "1";
	wander_step_duration = 0.02; wander_step_duration_in = "0.02";

	thresh_type = "infected"; thresh_type_in = "infected";
	testing_threshold_on = 0; testing_threshold_on_in = "0"; //  number of infected
	social_distance_threshold_on = 0; social_distance_threshold_on_in = "0"; //  number of hospitalized
	social_distance_threshold_off = 0; social_distance_threshold_off_in = "0"; //  number of remaining infected people
	social_distance_violation = 0; social_distance_violation_in = "0"; //  number of people
	SD_act_onset = false; SD_act_onset_in = "false";

	wall_buffer = 0.01; wall_buffer_in = "0.01"; // wall repulsion zone
	bounce_buffer = 0.005; bounce_buffer_in = "0.005";  // maximum overshoot outside wall

	// when people have an active destination, the wander range defines the area
	// surrounding the destination they will wander upon arriving
	wander_range = 0.05; wander_range_in = "0.05";
	wander_factor = 1; wander_factor_in = "1";
	wander_factor_dest = 1.5; wander_factor_dest_in = "1.5"; // area around destination

	// infection variables
	infection_range = 0.01; infection_range_in = "0.01"; // range surrounding sick patient that infections can take place
	infection_shape = "radial"; infection_shape_in = "radial"; // shape of infection zone surrounding sick patient that infections can take place
	infection_chance = 0.03; infection_chance_in = "0.03"; // chance that an infection spreads to nearby healthy people each tick
	recovery_duration = { 200, 500 }; recovery_duration_in = "200, 500"; // how many ticks it may take to recover from the illness
	mortality_chance = 0.02; mortality_chance_in = "0.02"; // global baseline chance of dying from the disease
	incubation_period = 0; incubation_period_in = "0"; // number of frames the individual spreads disease unknowingly
	patient_Z_loc = "random"; patient_Z_loc_in = "random";

	// healthcare variables
	healthcare_capacity = 300; healthcare_capacity_in = "300"; // capacity of the healthcare system
	treatment_factor = 0.5; treatment_factor_in = "0.5"; // when in treatment, affect risk by this factor
	no_treatment_factor = 3; no_treatment_factor_in = "3"; // risk increase factor to use if healthcare system is full
	// risk parameters
	treatment_dependent_risk = true; treatment_dependent_risk_in = "true"; // whether risk is affected by treatment

	// self isolation variables
	self_isolate_proportion = 0.6; self_isolate_proportion_in = "0.6";
	isolation_bounds = { 0.02, 0.02, 0.1, 0.98 }; isolation_bounds_in = "0.02, 0.02, 0.1, 0.98";
	number_of_tests = 10; number_of_tests_in = "10";

	// lockdown variables
	lockdown_percentage = 0.1; lockdown_percentage_in = "0.1";

	// random generator
	constant_seed = false; constant_seed_in = "false"; // whether user specifies a constant seed file or not
};

/*-----------------------------------------------------------*/
/*                      Throw exception                      */
/*-----------------------------------------------------------*/
void Configuration::config_error::throw_error(string key) 
{

	string s = "key " + key + " not present in config";
	throw s;

}

/*-----------------------------------------------------------*/
/*                        color palette                      */
/*-----------------------------------------------------------*/
vector<string> Configuration::get_palette() 
{

	/*returns appropriate color palette

	Uses config.plot_style to determine which palette to pick,
	and changes palette to colorblind mode(config.colorblind_mode)
	and colorblind type(config.colorblind_type) if required.

	Palette colors are based on
	https ://venngage.com/blog/color-blind-friendly-palette/

	*/

	// palette colors are : [healthy, infected, immune, dead]

	map< string, map<string, vector<string>> > palettes;

	map < string, vector<string> > regular;
	map < string, vector<string> > deuteranopia;
	map < string, vector<string> > protanopia;
	map < string, vector<string> > tritanopia;

	regular["dark"] = { "#1C758A", "#CF5044", "#BBBBBB", "#444444" };
	regular["default"] = { "#1C758A", "#CF5044", "#BBBBBB", "#444444" };

	deuteranopia["dark"] = { "gray", "#a50f15", "#08519c", "black" };
	deuteranopia["default"] = { "#404040", "#fcae91", "#6baed6", "#000000" };

	protanopia["dark"] = { "gray", "#a50f15", "08519c", "black" };
	protanopia["default"] = { "#404040", "#fcae91", "#6baed6", "#000000" };

	tritanopia["dark"] = { "gray", "#a50f15", "08519c", "black" };
	tritanopia["default"] = { "#404040", "#fcae91", "#6baed6", "#000000" };

	palettes["regular"] = regular;
	palettes["deuteranopia"] = deuteranopia;
	palettes["protanopia"] = protanopia;
	palettes["tritanopia"] = tritanopia;

	if (colorblind_mode) {
		return palettes[colorblind_type][plot_style];
	}
	else {
		return palettes["regular"][plot_style];
	}

}

/*-----------------------------------------------------------*/
/*                        set lockdown                       */
/*-----------------------------------------------------------*/
void Configuration::set_lockdown(RandomDevice *my_rand, double lockdown_percentage_var, double lockdown_compliance_var) 
{
	/*sets lockdown to active*/
	
	//fraction of the population that will obey the lockdown
	lockdown = true;
	lockdown_percentage = lockdown_percentage_var;

	//lockdown vector is 1 for those not complying
	lockdown_vector = my_rand->Random_choice_prob(pop_size, lockdown_compliance_var);

}

/*-----------------------------------------------------------*/
/*                  set self isolation ratio                 */
/*-----------------------------------------------------------*/
void Configuration::set_self_isolation(int number_of_tests_var, double self_isolate_proportion_var,
	vector<double> isolation_bounds_set, bool traveling_infects_var) 
{
	/*sets self-isolation scenario to active*/

	self_isolate = true;
	isolation_bounds = isolation_bounds_set;
	self_isolate_proportion = self_isolate_proportion_var;
	number_of_tests = number_of_tests_var; // careful not to confuse with objects ending in _in
	//set roaming bounds to outside isolated area
	xbounds = { 0.02, 0.98 };
	ybounds = { 0.02, 0.98 };
	//update plot bounds everything is shown
	x_plot = { isolation_bounds_set[0] - 0.02, 1 };
	y_plot = { 0, 1 };
	//update whether traveling agents also infect
	traveling_infects = traveling_infects_var;

}

/*-----------------------------------------------------------*/
/*          set lower speed for reduced interaction          */
/*-----------------------------------------------------------*/
void Configuration::set_reduced_interaction(double speed_var) 
{
	/*sets reduced interaction scenario to active*/

	speed = speed_var;

}

/*-----------------------------------------------------------*/
/*            Split delimited string into vector            */
/*-----------------------------------------------------------*/
vector<double> Configuration::split_string(string line) 
{
	string input = line;
	istringstream ss(input);
	string value;
	double value_db;

	vector<double> split_vector;
	while (getline(ss, value, ','))
	{
		value_db = atof(value.c_str()); // convert to float
		split_vector.push_back(value_db); // Vector of floats
		// cout << token << '\n';
	}


	size_t n_cols = split_vector.size();
	vector<double> output;
	for (size_t col_i = 0; col_i < n_cols; col_i++) {
		output.push_back(split_vector[col_i]);
	}

	return output;
}

/*-----------------------------------------------------------*/
/*          Set config values using external file            */
/*-----------------------------------------------------------*/
void Configuration::set_from_file() 
{
	pop_size = stoi(pop_size_in); // obtain pop_size first

	// these scaling factors where calculated for a default pop size of 2000
	area_scaling = 1.0 / (double(pop_size) / (600.0 / 2));
	distance_scaling = 1.0 / sqrt(double(pop_size) / (600.0 / 2));
	//distance_scaling = 1.0 / (double(pop_size) / (547.72));
	//distance_scaling = 1.0 / (double(pop_size) / (1000.0));
	force_scaling = pow(distance_scaling, 2) ;
	count_scaling = double(pop_size) / (600.0 / 2);

	// simulation variables
	simulation_steps = stoi(simulation_steps_in); // total simulation steps performed
	tstep = stoi(tstep_in); // current simulation timestep
	save_data = save_data_in == "true"; // whether to dump data at end of simulation
	save_pop = save_pop_in == "true"; // whether to save population matrix every "save_pop_freq" timesteps
	save_ground_covered = save_ground_covered_in == "true";  // whether to save ground covered matrix every "save_pop_freq" timesteps
	save_pop_freq = stoi(save_pop_freq_in); // population data will be saved every "n" timesteps.Default: 10
	save_pop_folder = save_pop_folder_in; // folder to write population timestep data to
	endif_no_infections = endif_no_infections_in == "true"; // whether to stop simulation if no infections remain
	write_bb_output = write_bb_output_in == "true"; // report results to black box output file for optimization
	// scenario flags
	traveling_infects = traveling_infects_in == "true";
	self_isolate = self_isolate_in == "true";
	lockdown = lockdown_in == "true";
	lockdown_compliance = stod(lockdown_compliance_in); // fraction of the population that will obey the lockdown

	// world variables, defines where population can and cannot roam
	xbounds = split_string(xbounds_in);
	ybounds = split_string(ybounds_in);
	n_gridpoints = stoi(n_gridpoints_in); //  resolution of 2D grid for tracking population position
	track_position = track_position_in == "true";
	track_GC = track_GC_in == "true";
	track_R0 = track_R0_in == "true";
	update_every_n_frame = stoi(update_every_n_frame_in);
	update_R0_every_n_frame = stoi(update_R0_every_n_frame_in);

	// visualisation variables
	visualise = visualise_in == "true"; // whether to visualise the simulation
	platform = platform_in; // which platform to use for visualization
	add_cross = add_cross_in == "true"; // draws a red cross on hospital
	visualise_every_n_frame = stoi(visualise_every_n_frame_in); // Frequency of plot update
	plot_mode = plot_mode_in; // default or sir
	n_plots = stoi(n_plots_in); // number of subplots
	plot_last_tstep = plot_last_tstep_in == "true"; // plot last frame SIR
	trace_path = trace_path_in == "true"; // trace path of a single individual
	// size of the simulated world in coordinates
	x_plot = split_string(x_plot_in);
	y_plot = split_string(y_plot_in);
	save_plot = save_plot_in == "true";
	plot_path = plot_path_in; // folder where plots are saved to
	plot_style = plot_style_in; // can be default, dark, ...
	plot_text_style = plot_text_style_in; // can be default, LaTeX, ...
	colorblind_mode = colorblind_mode_in == "true";
	// if colorblind is enabled, set type of colorblindness
	// available: deuteranopia, protanopia, tritanopia.defauld = deuteranopia
	colorblind_type = colorblind_type_in;
	verbose = verbose_in == "true"; // output stats to console
	report_freq = stoi(report_freq_in); // report results every frame to console
	report_status = report_status_in == "true"; // output stats to console
	marker_size = stoi(marker_size_in); // markersize for plotting individuals

	// population variables
	mean_age = stoi(mean_age_in);
	max_age = stoi(max_age_in);
	age_dependent_risk = age_dependent_risk_in == "true"; // whether risk increases with age
	risk_age = stoi(risk_age_in); // age where mortality risk starts increasing
	critical_age = stoi(critical_age_in); // age at and beyond which mortality risk reaches maximum
	critical_mortality_chance = stod(critical_mortality_chance_in); // maximum mortality risk for older age
	risk_increase = risk_increase_in; // whether risk between risk and critical age increases "linear" or "quadratic"

	// movement variables
	// the proportion of the population that practices social distancing, simulated
	// by them standing still
	proportion_distancing = stoi(proportion_distancing_in);
	social_distance_factor = 1e-6 * stod(social_distance_factor_in) * force_scaling;
	speed = stod(speed_in) * distance_scaling; // average speed of population
	max_speed = stod(max_speed_in) * distance_scaling; // average speed of population
	dt = stod(dt_in); // average speed of population

	wander_step_size = stod(wander_step_size_in) * distance_scaling;
	gravity_strength = stod(gravity_strength_in);
	wander_step_duration = stod(wander_step_duration_in);

	thresh_type = thresh_type_in;
	testing_threshold_on = stoi(testing_threshold_on_in); // number of infected
	social_distance_threshold_on = stoi(social_distance_threshold_on_in); // number of hospitalized
	social_distance_threshold_off = stoi(social_distance_threshold_off_in); // number of remaining infected people
	social_distance_violation = stoi(social_distance_violation_in); // number of people
	SD_act_onset = SD_act_onset_in == "true";

	wall_buffer = stod(wall_buffer_in); // wall repulsion zone
	bounce_buffer = stod(bounce_buffer_in) * distance_scaling; // maximum overshoot outside wall

	// when people have an active destination, the wander range defines the area
	// surrounding the destination they will wander upon arriving
	wander_range = stod(wander_range_in);
	wander_factor = stod(wander_factor_in);
	wander_factor_dest = stod(wander_factor_dest_in); // area around destination

	// infection variables
	infection_range = stod(infection_range_in) * distance_scaling; // range surrounding sick patient that infections can take place
	infection_shape = infection_shape_in; // shape of infection zone surrounding sick patient that infections can take place
	infection_chance =stod(infection_chance_in);   // chance that an infection spreads to nearby healthy people each tick
	recovery_duration = { 200, 500 }; // how many ticks it may take to recover from the illness
	mortality_chance = stod(mortality_chance_in); // global baseline chance of dying from the disease
	incubation_period = stoi(incubation_period_in); // number of frames the individual spreads disease unknowingly
	patient_Z_loc = patient_Z_loc_in;

	// healthcare variables
	healthcare_capacity = stoi(healthcare_capacity_in); // capacity of the healthcare system
	treatment_factor = stod(treatment_factor_in); // when in treatment, affect risk by this factor
	no_treatment_factor = stod(no_treatment_factor_in); // risk increase factor to use if healthcare system is full
	// risk parameters
	treatment_dependent_risk = treatment_dependent_risk_in == "true"; // whether risk is affected by treatment

	// self isolation variables
	self_isolate_proportion = stod(self_isolate_proportion_in);
	isolation_bounds = split_string(isolation_bounds_in);
	number_of_tests = stoi(number_of_tests_in);
	if (self_isolate) {x_plot[0] = isolation_bounds[0] - 0.02;}

	// lockdown variables
	lockdown_percentage = stod(lockdown_percentage_in);

	// random generator
	constant_seed = constant_seed_in == "true"; // whether user specifies a constant seed file or not

}

/*-----------------------------------------------------------*/
/*                        Destructor                         */
/*-----------------------------------------------------------*/
Configuration::~Configuration()
{
}

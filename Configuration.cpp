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
	simulation_steps = 10000; // total simulation steps performed
	tstep = 0; // current simulation timestep
	save_data = false; // whether to dump data at end of simulation
	save_pop = false; // whether to save population matrix every "save_pop_freq" timesteps
	save_pop_freq = 10; // population data will be saved every "n" timesteps.Default: 10
	save_pop_folder = "population"; // folder to write population timestep data to
	endif_no_infections = true; // whether to stop simulation if no infections remain

	// scenario flags
	traveling_infects = false;
	self_isolate = false;
	lockdown = false;
	lockdown_compliance = 0.95; // fraction of the population that will obey the lockdown

	// world variables, defines where population can and cannot roam
	xbounds = { 0.02, 0.98 };
	ybounds = { 0.02, 0.98 };
	n_gridpoints = 10; //  resolution of 2D grid for tracking population position
	track_position = true;
	track_GC = false;
	update_every_n_frame = 1;

	// visualisation variables
	visualise = true; // whether to visualise the simulation
	add_cross = false; // draws a red cross on hospital
	visualise_every_n_frame = 1; // Frequency of plot update
	plot_mode = "sir"; // default or sir
	n_plots = 2; // number of subplots
	plot_last_tstep = true; // plot last frame SIR
	trace_path = false; // trace path of a single individual
	// size of the simulated world in coordinates
	x_plot = { 0, 1 };
	y_plot = { 0, 1 };
	save_plot = false;
	plot_path = "render"; // folder where plots are saved to
	plot_style = "default"; // can be default, dark, ...
	plot_text_style = "default"; // can be default, LaTeX, ...
	colorblind_mode = false;
	// if colorblind is enabled, set type of colorblindness
	// available: deuteranopia, protanopia, tritanopia.defauld = deuteranopia
	colorblind_type = "deuteranopia";
	verbose = true; // output stats to console
	report_freq = 1; // report results every frame to console
	report_status = false; // output stats to console
	marker_size = 15; // markersize for plotting individuals

	// population variables
	pop_size = 2000;
	mean_age = 45;
	max_age = 105;
	age_dependent_risk = true; // whether risk increases with age
	risk_age = 55; // age where mortality risk starts increasing
	critical_age = 75; // age at and beyond which mortality risk reaches maximum
	critical_mortality_chance = 0.1; // maximum mortality risk for older age
	risk_increase = "quadratic"; // whether risk between risk and critical age increases "linear" or "quadratic"

	// movement variables
	// the proportion of the population that practices social distancing, simulated
	// by them standing still
	proportion_distancing = 0;
	social_distance_factor = 0.0;
	speed = 0.01; // average speed of population
	max_speed = 1.0; // average speed of population
	dt = 0.01; // average speed of population

	wander_step_size = 0.01;
	gravity_strength = 1;
	wander_step_duration = 0.02;

	thresh_type = "infected";
	testing_threshold_on = 0; //  number of infected
	social_distance_threshold_on = 0; //  number of hospitalized
	social_distance_threshold_off = 0; //  number of remaining infected people
	social_distance_violation = 0; //  number of people
	SD_act_onset = false;

	// when people have an active destination, the wander range defines the area
	// surrounding the destination they will wander upon arriving
	wander_range = 0.05;
	wander_factor = 1;
	wander_factor_dest = 1.5; // area around destination

	// infection variables
	infection_range = 0.01; // range surrounding sick patient that infections can take place
	infection_chance = 0.03;   // chance that an infection spreads to nearby healthy people each tick
	recovery_duration = { 200, 500 }; // how many ticks it may take to recover from the illness
	mortality_chance = 0.02; // global baseline chance of dying from the disease
	incubation_period = 0; // number of frames the individual spreads disease unknowingly
	patient_Z_loc = "random";

	// healthcare variables
	healthcare_capacity = 300; // capacity of the healthcare system
	treatment_factor = 0.5; // when in treatment, affect risk by this factor
	no_treatment_factor = 3; // risk increase factor to use if healthcare system is full
	// risk parameters
	treatment_dependent_risk = true; // whether risk is affected by treatment

	// self isolation variables
	self_isolate_proportion = 0.6;
	isolation_bounds = { 0.02, 0.02, 0.1, 0.98 };
	number_of_tests = 10;

	// lockdown variables
	lockdown_percentage = 0.1;
	lockdown_vector;
};

/*-----------------------------------------------------------*/
/*                      Throw exception                      */
/*-----------------------------------------------------------*/
void Configuration::config_error::throw_error(string key) {

	string s = "key " + key + " not present in config";
	throw s;

}
/*-----------------------------------------------------------*/
/*                        color palette                      */
/*-----------------------------------------------------------*/
vector<string> Configuration::get_palette() {

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

};

/*-----------------------------------------------------------*/
/*                        set lockdown                       */
/*-----------------------------------------------------------*/
void Configuration::set_lockdown(RandomDevice *my_rand, double lockdown_percentage_in, double lockdown_compliance_in) {
	/*sets lockdown to active*/
	
	//fraction of the population that will obey the lockdown
	lockdown = true;
	lockdown_percentage = lockdown_percentage_in;

	//lockdown vector is 1 for those not complying
	lockdown_vector = my_rand->Random_choice_prob(pop_size, lockdown_compliance_in);

};

/*-----------------------------------------------------------*/
/*                  set self isolation ratio                 */
/*-----------------------------------------------------------*/
void Configuration::set_self_isolation(int number_of_tests_in, double self_isolate_proportion_in,
	vector<double> isolation_bounds_in,
	bool traveling_infects_in) {
	/*sets self-isolation scenario to active*/

	self_isolate = true;
	isolation_bounds = isolation_bounds_in;
	self_isolate_proportion = self_isolate_proportion_in;
	number_of_tests = number_of_tests_in;
	//set roaming bounds to outside isolated area
	xbounds = { 0.02, 0.98 };
	ybounds = { 0.02, 0.98 };
	//update plot bounds everything is shown
	x_plot = { isolation_bounds_in[0] - 0.02, 1 };
	y_plot = { 0, 1 };
	//update whether traveling agents also infect
	traveling_infects = traveling_infects_in;

};

/*-----------------------------------------------------------*/
/*          set lower speed for reduced interaction          */
/*-----------------------------------------------------------*/
void Configuration::set_reduced_interaction(double speed_in) {
	/*sets reduced interaction scenario to active*/

	speed = speed_in;

};

/*-----------------------------------------------------------*/
/*                        Destructor                         */
/*-----------------------------------------------------------*/
Configuration::~Configuration()
{
}

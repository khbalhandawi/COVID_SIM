#pragma once

#include "RandomDevice.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

class Configuration
{
public:

	/*-----------------------------------------------------------*/
	/*            Simulation configuration variables             */
	/*-----------------------------------------------------------*/

	// simulation variables
	int simulation_steps; // total simulation steps performed
	int tstep; // current simulation timestep
	bool save_data; // whether to dump data at end of simulation
	bool save_pop; // whether to save population matrix every "save_pop_freq" timesteps
	int save_pop_freq; // population data will be saved every "n" timesteps.Default: 10
	string save_pop_folder; // folder to write population timestep data to
	bool endif_no_infections; // whether to stop simulation if no infections remain

	// scenario flags
	bool traveling_infects;
	bool self_isolate;
	bool lockdown;
	double lockdown_compliance; // fraction of the population that will obey the lockdown

	// world variables, defines where population can and cannot roam
	vector<double> xbounds;
	vector<double> ybounds;
	int n_gridpoints = 10; //  resolution of 2D grid for tracking population position
	bool track_position;
	bool track_GC;
	int update_every_n_frame;

	// visualisation variables
	bool visualise; // whether to visualise the simulation
	bool add_cross; // draws a red cross on hospital
	int visualise_every_n_frame; // Frequency of plot update
	string plot_mode; // default or sir
	int n_plots; // 1 or 2
	bool plot_last_tstep; // plot last frame SIR
	bool trace_path; // trace path of a single individual
	// size of the simulated world in coordinates
	vector<double> x_plot;
	vector<double> y_plot;
	bool save_plot = false;
	string plot_path; // folder where plots are saved to
	string plot_style; // can be default, dark, ...
	string plot_text_style; // can be default, LaTeX, ...
	bool colorblind_mode;
	// if colorblind is enabled, set type of colorblindness
	// available: deuteranopia, protanopia, tritanopia.defauld = deuteranopia
	string colorblind_type;
	bool verbose; // output stats to console
	int report_freq; // report stats to console every n frames
	bool report_status; // output stats to console
	double marker_size; // markersize for plotting individuals

	// population variables
	int pop_size;
	int mean_age;
	int max_age;
	bool age_dependent_risk; // whether risk increases with age
	int risk_age; // age where mortality risk starts increasing
	int critical_age; // age at and beyond which mortality risk reaches maximum
	double critical_mortality_chance; // maximum mortality risk for older age
	string risk_increase; // whether risk between risk and critical age increases "linear" or "quadratic"

	// movement variables
	// mean_speed = 0.01 //  the mean speed(defined as heading * speed)
	// std_speed = 0.01 / 3 // the standard deviation of the speed parameter
	// the proportion of the population that practices social distancing, simulated
	// by them standing still
	int proportion_distancing;
	double social_distance_factor;
	double speed; // average speed of population
	double max_speed; // average speed of population
	double dt; // average speed of population

	double wander_step_size;
	double gravity_strength;
	double wander_step_duration;

	string thresh_type = "infected";
	int testing_threshold_on; //  number of infected
	int social_distance_threshold_on; //  number of hospitalized
	int social_distance_threshold_off; //  number of remaining infected people
	int social_distance_violation; //  number of people
	bool SD_act_onset;

	// when people have an active destination, the wander range defines the area
	// surrounding the destination they will wander upon arriving
	double wander_range;
	double wander_factor;
	double wander_factor_dest; // area around destination

	// infection variables
	double infection_range; // range surrounding sick patient that infections can take place
	double infection_chance;   // chance that an infection spreads to nearby healthy people each tick
	vector<int> recovery_duration; // how many ticks it may take to recover from the illness
	double mortality_chance; // global baseline chance of dying from the disease
	int incubation_period; // number of frames the individual spreads disease unknowingly
	string patient_Z_loc;

	// healthcare variables
	int healthcare_capacity; // capacity of the healthcare system
	double treatment_factor; // when in treatment, affect risk by this factor
	double no_treatment_factor; // risk increase factor to use if healthcare system is full
	// risk parameters
	bool treatment_dependent_risk; // whether risk is affected by treatment

	// self isolation variables
	double self_isolate_proportion;
	vector<double> isolation_bounds;
	int number_of_tests; // number of people tested

	// lockdown variables
	double lockdown_percentage;
	Eigen::ArrayXf lockdown_vector;

	/*-----------------------------------------------------------*/
	/*          Throw exception if config key not found          */
	/*-----------------------------------------------------------*/
	class config_error {
	public:
		void throw_error(string key);
	};

	/*-----------------------------------------------------------*/
	/*                        color palette                      */
	/*-----------------------------------------------------------*/
	vector<string> get_palette();

	/*-----------------------------------------------------------*/
	/*                        set lockdown                       */
	/*-----------------------------------------------------------*/
	void set_lockdown(RandomDevice *my_rand, double lockdown_percentage = 0.1, double lockdown_compliance = 0.9);
	
	/*-----------------------------------------------------------*/
	/*                  set self isolation ratio                 */
	/*-----------------------------------------------------------*/
	void set_self_isolation(int number_of_tests, double self_isolate_proportion = 0.9,
		vector<double> isolation_bounds = { -0.28, 0.02, -0.02, 0.28 },
		bool traveling_infects = false);

	/*-----------------------------------------------------------*/
	/*          set lower speed for reduced interaction          */
	/*-----------------------------------------------------------*/
	void set_reduced_interaction(double speed = 0.001);

	/*-----------------------------------------------------------*/
	/*                        Constructor                        */
	/*-----------------------------------------------------------*/
	Configuration();

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~Configuration();
};
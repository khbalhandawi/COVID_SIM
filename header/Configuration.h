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
	/*                     Config file values                    */
	/*-----------------------------------------------------------*/

	// simulation variables
	string simulation_steps_in; // total simulation steps performed
	string tstep_in; // current simulation timestep
	string save_data_in; // whether to dump data at end of simulation
	string save_pop_in; // whether to save population matrix every "save_pop_freq" timesteps
	string save_ground_covered_in; // whether to save ground covered matrix every "save_pop_freq" timesteps
	string save_pop_freq_in; // population data will be saved every "n" timesteps.Default: 10
	string save_pop_folder_in; // folder to write population timestep data to
	string endif_no_infections_in; // whether to stop simulation if no infections remain
	string write_bb_output_in; // report results to black box output file for optimization
	// scenario flags
	string traveling_infects_in;
	string self_isolate_in;
	string lockdown_in;
	string lockdown_compliance_in; // fraction of the population that will obey the lockdown

	// world variables, defines where population can and cannot roam
	string xbounds_in;
	string ybounds_in;
	string n_gridpoints_in; //  resolution of 2D grid for tracking population position
	string track_position_in;
	string track_GC_in;
	string track_R0_in;
	string update_every_n_frame_in;
	string update_R0_every_n_frame_in;

	// visualisation variables
	string visualise_in; // whether to visualise the simulation
	string platform_in; // which platform to use for visualization
	string add_cross_in; // draws a red cross on hospital
	string visualise_every_n_frame_in; // Frequency of plot update
	string plot_mode_in; // default or sir
	string n_plots_in; // 1 or 2
	string plot_last_tstep_in; // plot last frame SIR
	string trace_path_in; // trace path of a single individual
	// size of the simulated world in coordinates
	string x_plot_in;
	string y_plot_in;
	string save_plot_in;
	string plot_path_in; // folder where plots are saved to
	string plot_style_in; // can be default, dark, ...
	string plot_text_style_in; // can be default, LaTeX, ...
	string colorblind_mode_in;
	// if colorblind is enabled, set type of colorblindness
	// available: deuteranopia, protanopia, tritanopia.defauld = deuteranopia
	string colorblind_type_in;
	string verbose_in; // output stats to console
	string report_freq_in; // report stats to console every n frames
	string report_status_in; // output stats to console
	string marker_size_in; // markersize for plotting individuals

	// population variables
	string pop_size_in;
	string mean_age_in;
	string max_age_in;
	string age_dependent_risk_in; // whether risk increases with age
	string risk_age_in; // age where mortality risk starts increasing
	string critical_age_in; // age at and beyond which mortality risk reaches maximum
	string critical_mortality_chance_in; // maximum mortality risk for older age
	string risk_increase_in; // whether risk between risk and critical age increases "linear" or "quadratic"

	// movement variables
	// mean_speed = 0.01 //  the mean speed(defined as heading * speed)
	// std_speed = 0.01 / 3 // the standard deviation of the speed parameter
	// the proportion of the population that practices social distancing, simulated
	// by them standing still
	string proportion_distancing_in;
	string social_distance_factor_in;
	string speed_in; // average speed of population
	string max_speed_in; // average speed of population
	string dt_in; // average speed of population

	string wander_step_size_in;
	string gravity_strength_in;
	string wander_step_duration_in;

	string thresh_type_in;
	string testing_threshold_on_in; //  number of infected
	string social_distance_threshold_on_in; //  number of hospitalized
	string social_distance_threshold_off_in; //  number of remaining infected people
	string social_distance_violation_in; //  number of people
	string SD_act_onset_in;

	// when people have an active destination, the wander range defines the area
	// surrounding the destination they will wander upon arriving
	string wander_range_in;
	string wander_factor_in;
	string wander_factor_dest_in; // area around destination

	// infection variables
	string infection_range_in; // range surrounding sick patient that infections can take place
	string infection_shape_in; // shape of infection zone surrounding sick patient that infections can take place
	string infection_chance_in;   // chance that an infection spreads to nearby healthy people each tick
	string recovery_duration_in; // how many ticks it may take to recover from the illness
	string mortality_chance_in; // global baseline chance of dying from the disease
	string incubation_period_in; // number of frames the individual spreads disease unknowingly
	string patient_Z_loc_in;

	// healthcare variables
	string healthcare_capacity_in; // capacity of the healthcare system
	string treatment_factor_in; // when in treatment, affect risk by this factor
	string no_treatment_factor_in; // risk increase factor to use if healthcare system is full
	// risk parameters
	string treatment_dependent_risk_in; // whether risk is affected by treatment

	// self isolation variables
	string self_isolate_proportion_in;
	string isolation_bounds_in;
	string number_of_tests_in; // number of people tested

	// lockdown variables
	string lockdown_percentage_in;

	// random generator
	string constant_seed_in; // whether user specifies a constant seed file or not

	/*-----------------------------------------------------------*/
	/*            Simulation configuration variables             */
	/*-----------------------------------------------------------*/

	// simulation variables
	int simulation_steps; // total simulation steps performed
	int tstep; // current simulation timestep
	bool save_data; // whether to dump data at end of simulation
	bool save_pop; // whether to save population matrix every "save_pop_freq" timesteps
	bool save_ground_covered; // whether to save ground covered matrix every "save_pop_freq" timesteps
	int save_pop_freq; // population data will be saved every "n" timesteps.Default: 10
	string save_pop_folder; // folder to write population timestep data to
	bool endif_no_infections; // whether to stop simulation if no infections remain
	bool write_bb_output; // report results to black box output file for optimization
	// scenario flags
	bool traveling_infects;
	bool self_isolate;
	bool lockdown;
	double lockdown_compliance; // fraction of the population that will obey the lockdown

	// world variables, defines where population can and cannot roam
	vector<double> xbounds;
	vector<double> ybounds;
	int n_gridpoints; //  resolution of 2D grid for tracking population position
	bool track_position;
	bool track_GC;
	bool track_R0;
	int update_every_n_frame;
	int update_R0_every_n_frame;

	// visualisation variables
	bool visualise; // whether to visualise the simulation
	string platform; // which platform to use for visualization
	bool add_cross; // draws a red cross on hospital
	int visualise_every_n_frame; // Frequency of plot update
	string plot_mode; // default or sir
	int n_plots; // 1 or 2
	bool plot_last_tstep; // plot last frame SIR
	bool trace_path; // trace path of a single individual
	// size of the simulated world in coordinates
	vector<double> x_plot;
	vector<double> y_plot;
	bool save_plot;
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

	string thresh_type;
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
	string infection_shape; // shape of infection zone surrounding sick patient that infections can take place
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

	// random generator
	bool constant_seed; // whether user specifies a constant seed file or not

	Eigen::ArrayXf lockdown_vector;

	// Scaling factors
	double area_scaling;
	double distance_scaling;
	double force_scaling;
	double count_scaling;

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
	void set_lockdown(RandomDevice *my_rand, double lockdown_percentage_var = 0.1, double lockdown_compliance_var = 0.9);
	
	/*-----------------------------------------------------------*/
	/*                  set self isolation ratio                 */
	/*-----------------------------------------------------------*/
	void set_self_isolation(int number_of_tests_var, double self_isolate_proportion_var = 0.9,
		vector<double> isolation_bounds_var = { -0.28, 0.02, -0.02, 0.28 },
		bool traveling_infects_var = false);

	/*-----------------------------------------------------------*/
	/*          set lower speed for reduced interaction          */
	/*-----------------------------------------------------------*/
	void set_reduced_interaction(double speed_var = 0.001);

	/*-----------------------------------------------------------*/
	/*            Split delimlited string into vector            */
	/*-----------------------------------------------------------*/
	vector<double> split_string(string line);

	/*-----------------------------------------------------------*/
	/*          Set config values using external file            */
	/*-----------------------------------------------------------*/
	void set_from_file();

	/*-----------------------------------------------------------*/
	/*                        Constructor                        */
	/*-----------------------------------------------------------*/
	Configuration();

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~Configuration();
};
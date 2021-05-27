#pragma once

#include "RandomDevice.h"
#include "Defines.h"

#include<vector>
#include<string>
#include<Eigen/Core>

#ifndef CONFIGURATION_H_H
#define CONFIGURATION_H_H

namespace COVID_SIM {

MACOS_API_BEGIN
	class DLL_API Configuration
	{
	public:

		/*-----------------------------------------------------------*/
		/*                     Config file values                    */
		/*-----------------------------------------------------------*/

		// simulation variables
		std::string simulation_steps_in; // total simulation steps performed
		std::string tstep_in; // current simulation timestep
		std::string save_data_in; // whether to dump data at end of simulation
		std::string save_pop_in; // whether to save population matrix every "save_pop_freq" timesteps
		std::string save_ground_covered_in; // whether to save ground covered matrix every "save_pop_freq" timesteps
		std::string save_pop_freq_in; // population data will be saved every "n" timesteps.Default: 10
		std::string save_pop_folder_in; // folder to write population timestep data to
		std::string endif_no_infections_in; // whether to stop simulation if no infections remain
		std::string write_bb_output_in; // report results to black box output file for optimization
		// scenario flags
		std::string traveling_infects_in;
		std::string self_isolate_in;
		std::string lockdown_in;
		std::string lockdown_compliance_in; // fraction of the population that will obey the lockdown

		// world variables, defines where population can and cannot roam
		std::string xbounds_in;
		std::string ybounds_in;
		std::string n_gridpoints_in; //  resolution of 2D grid for tracking population position
		std::string track_position_in;
		std::string track_GC_in;
		std::string track_R0_in;
		std::string update_every_n_frame_in;
		std::string update_R0_every_n_frame_in;

		// visualisation variables
		std::string visualise_in; // whether to visualise the simulation
		std::string platform_in; // which platform to use for visualization
		std::string add_cross_in; // draws a red cross on hospital
		std::string visualise_every_n_frame_in; // Frequency of plot update
		std::string plot_mode_in; // default or sir
		std::string n_plots_in; // 1 or 2
		std::string plot_last_tstep_in; // plot last frame SIR
		std::string trace_path_in; // trace path of a single individual
		// size of the simulated world in coordinates
		std::string x_plot_in;
		std::string y_plot_in;
		std::string save_plot_in;
		std::string plot_path_in; // folder where plots are saved to
		std::string plot_style_in; // can be default, dark, ...
		std::string plot_text_style_in; // can be default, LaTeX, ...
		std::string colorblind_mode_in;
		// if colorblind is enabled, set type of colorblindness
		// available: deuteranopia, protanopia, tritanopia.defauld = deuteranopia
		std::string colorblind_type_in;
		std::string verbose_in; // output stats to console
		std::string report_freq_in; // report stats to console every n frames
		std::string report_status_in; // output stats to console
		std::string marker_size_in; // markersize for plotting individuals

		// population variables
		std::string pop_size_in;
		std::string mean_age_in;
		std::string max_age_in;
		std::string age_dependent_risk_in; // whether risk increases with age
		std::string risk_age_in; // age where mortality risk starts increasing
		std::string critical_age_in; // age at and beyond which mortality risk reaches maximum
		std::string critical_mortality_chance_in; // maximum mortality risk for older age
		std::string risk_increase_in; // whether risk between risk and critical age increases "linear" or "quadratic"

		// movement variables
		// mean_speed = 0.01 //  the mean speed(defined as heading * speed)
		// std_speed = 0.01 / 3 // the standard deviation of the speed parameter
		// the proportion of the population that practices social distancing, simulated
		// by them standing still
		std::string proportion_distancing_in;
		std::string social_distance_factor_in;
		std::string speed_in; // average speed of population
		std::string max_speed_in; // average speed of population
		std::string dt_in; // average speed of population

		std::string wander_step_size_in;
		std::string gravity_strength_in;
		std::string wander_step_duration_in;

		std::string thresh_type_in;
		std::string testing_threshold_on_in; //  number of infected
		std::string social_distance_threshold_on_in; //  number of hospitalized
		std::string social_distance_threshold_off_in; //  number of remaining infected people
		std::string social_distance_violation_in; //  number of people
		std::string SD_act_onset_in;

		std::string wall_buffer_in; // wall repulsion zone
		std::string bounce_buffer_in;  // maximum overshoot outside wall

		// when people have an active destination, the wander range defines the area
		// surrounding the destination they will wander upon arriving
		std::string wander_range_in;
		std::string wander_factor_in;
		std::string wander_factor_dest_in; // area around destination

		// infection variables
		std::string infection_range_in; // range surrounding sick patient that infections can take place
		std::string infection_shape_in; // shape of infection zone surrounding sick patient that infections can take place
		std::string infection_chance_in;   // chance that an infection spreads to nearby healthy people each tick
		std::string recovery_duration_in; // how many ticks it may take to recover from the illness
		std::string mortality_chance_in; // global baseline chance of dying from the disease
		std::string incubation_period_in; // number of frames the individual spreads disease unknowingly
		std::string patient_Z_loc_in;  // Where to seed the infection (center of box, or randomly inside box)
		std::string patient_Z_time_in; // time at which to seed the infection

		// healthcare variables
		std::string healthcare_capacity_in; // capacity of the healthcare system
		std::string treatment_factor_in; // when in treatment, affect risk by this factor
		std::string no_treatment_factor_in; // risk increase factor to use if healthcare system is full
		// risk parameters
		std::string treatment_dependent_risk_in; // whether risk is affected by treatment

		// self isolation variables
		std::string self_isolate_proportion_in;
		std::string isolation_bounds_in;
		std::string number_of_tests_in; // number of people tested

		// lockdown variables
		std::string lockdown_percentage_in;

		// random generator
		std::string constant_seed_in; // whether user specifies a constant seed file or not

		// Export settings
		std::string log_file_in; // blackbox history file
		std::string run_i_in; // blackbox run ID

		// UI settings
		std::string IC_max_in;// maximum slider position
		std::string IC_min_in; // minimum slider position
		std::string SD_max_in;// maximum slider position
		std::string SD_min_in; // minimum slider position
		std::string TC_max_in; // maximum slider position
		std::string TC_min_in; // minimum slider position

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
		std::string save_pop_folder; // folder to write population timestep data to
		bool endif_no_infections; // whether to stop simulation if no infections remain
		bool write_bb_output; // report results to black box output file for optimization
		// scenario flags
		bool traveling_infects;
		bool self_isolate;
		bool lockdown;
		double lockdown_compliance; // fraction of the population that will obey the lockdown

		// world variables, defines where population can and cannot roam
		std::vector<double> xbounds;
		std::vector<double> ybounds;
		int n_gridpoints; //  resolution of 2D grid for tracking population position
		bool track_position;
		bool track_GC;
		bool track_R0;
		int update_every_n_frame;
		int update_R0_every_n_frame;

		// visualisation variables
		bool visualise; // whether to visualise the simulation
		std::string platform; // which platform to use for visualization
		bool add_cross; // draws a red cross on hospital
		int visualise_every_n_frame; // Frequency of plot update
		std::string plot_mode; // default or sir
		int n_plots; // 1 or 2
		bool plot_last_tstep; // plot last frame SIR
		bool trace_path; // trace path of a single individual
		// size of the simulated world in coordinates
		std::vector<double> x_plot;
		std::vector<double> y_plot;
		bool save_plot;
		std::string plot_path; // folder where plots are saved to
		std::string plot_style; // can be default, dark, ...
		std::string plot_text_style; // can be default, LaTeX, ...
		bool colorblind_mode;
		// if colorblind is enabled, set type of colorblindness
		// available: deuteranopia, protanopia, tritanopia.defauld = deuteranopia
		std::string colorblind_type;
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
		std::string risk_increase; // whether risk between risk and critical age increases "linear" or "quadratic"

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

		std::string thresh_type;
		int testing_threshold_on; //  number of infected
		int social_distance_threshold_on; //  number of hospitalized
		int social_distance_threshold_off; //  number of remaining infected people
		int social_distance_violation; //  number of people
		bool SD_act_onset;

		double wall_buffer; // wall repulsion zone
		double bounce_buffer;  // maximum overshoot outside wall

		// when people have an active destination, the wander range defines the area
		// surrounding the destination they will wander upon arriving
		double wander_range;
		double wander_factor;
		double wander_factor_dest; // area around destination

		// infection variables
		double infection_range; // range surrounding sick patient that infections can take place
		std::string infection_shape; // shape of infection zone surrounding sick patient that infections can take place
		double infection_chance;   // chance that an infection spreads to nearby healthy people each tick
		std::vector<int> recovery_duration; // how many ticks it may take to recover from the illness
		double mortality_chance; // global baseline chance of dying from the disease
		int incubation_period; // number of frames the individual spreads disease unknowingly
		std::string patient_Z_loc;  // Where to seed the infection (center of box, or randomly inside box)
		int patient_Z_time; // time at which to seed the infection

		// healthcare variables
		int healthcare_capacity; // capacity of the healthcare system
		double treatment_factor; // when in treatment, affect risk by this factor
		double no_treatment_factor; // risk increase factor to use if healthcare system is full
		// risk parameters
		bool treatment_dependent_risk; // whether risk is affected by treatment

		// self isolation variables
		double self_isolate_proportion;
		std::vector<double> isolation_bounds;
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

		// Export settings
		std::string log_file; // blackbox history file
		int run_i; // blackbox run ID

		// UI settings
		double IC_max; // maximum slider position
		double IC_min; // minimum slider position
		double SD_max; // maximum slider position
		double SD_min; // minimum slider position
		double TC_max; // maximum slider position
		double TC_min; // minimum slider position

		/*-----------------------------------------------------------*/
		/*          Throw exception if config key not found          */
		/*-----------------------------------------------------------*/
		class config_error {
		public:
			void throw_error(std::string key);
		};

		/*-----------------------------------------------------------*/
		/*                        color palette                      */
		/*-----------------------------------------------------------*/
		std::vector<std::string> get_palette();

		/*-----------------------------------------------------------*/
		/*                        set lockdown                       */
		/*-----------------------------------------------------------*/
		void set_lockdown(RandomDevice *my_rand, double lockdown_percentage_var = 0.1, double lockdown_compliance_var = 0.9);

		/*-----------------------------------------------------------*/
		/*          set lower speed for reduced interaction          */
		/*-----------------------------------------------------------*/
		void set_reduced_interaction(double speed_var = 0.001);

		/*-----------------------------------------------------------*/
		/*            Split delimlited string into vector            */
		/*-----------------------------------------------------------*/
		std::vector<double> split_string(std::string line);

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
MACOS_API_END

}

#endif // CONFIGURATION_H_H

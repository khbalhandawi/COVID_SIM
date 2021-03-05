# List of configuration parameters

Below is the list of model parameters and their possible values for editing the files [configuration_debug.ini](build/configuration_debug.ini) and [configuration.ini](build/configuration.ini). Please use the same syntax used in these files when defining parameters and pay attention to spaces and commas used.

Parameter	     				| Possible values 		| Default  	    | Description
:------------------------------:|:---------------------:|:-------------:|:------------:
simulation_steps 				| > 0					| 10000 		| total simulation steps performed
save_data 		 				| false,true 			| false 		| whether to dump data at end of simulation
save_pop 		 				| false,true 			| false 		| whether to save population matrix every "save_pop_freq" timesteps
save_ground_covered 			| false,true 			| false 		| whether to save ground covered matrix every "save_pop_freq" timesteps
save_pop_freq 					| > 0					| 10 			| population data will be saved every "n" timesteps
save_pop_folder 				| any string 			| population 	| folder to write population timestep data to
endif_no_infections 			| false,true			| true			| whether to stop simulation if no infections remain
write_bb_output 				| false,true			| true			| report results to black box output file for optimization
**scenario flags**				|						|				|
traveling_infects 				| false,true 			| false 		|
self_isolate					| false,true 			| false 		| whether a quarantine environment will be used
**world**  						|						|				|
xbounds							| > 0, > 0 				| 0.02, 0.98 	| lower and upper x bounds of the "world" environment
ybounds							| > 0, > 0 				| 0.02, 0.98 	| lower and upper y bounds of the "world" environment
n_gridpoints 					| > 0 					| 10 			| resolution of 2D grid for tracking population position
track_position 					| false,true			| true			| track the total distance travelled by agents
track_GC 						| false,true			| false			| track the mobility of agents (computationally expensive - best with GPU)
track_R0 						| false,true			| false			| track the basic reproductive number
update_every_n_frame 			| > 0					| 1				| mobility will be calculated every "n" timesteps
update_R0_every_n_frame 		| > 0					| 1				| reproductive number will be calculated every "n" timesteps
**visualisation** 				|						|				|
visualise 						| false,true			| true			| whether to visualize the simulation
platform 						| Qt					| true			| which platform to use for visualization
visualise_every_n_frame 		| > 0					| 1				| Frequency of plot update
n_plots 						| 1, 2					| 2				| number of subplots (if 1, SIR plot will be hidden)
trace_path 						| false,true			| false			| trace path of a single individual
save_plot 						| false,true			| false			| whether to save plots of simulation
plot_path 						| any string			| render		| directory in which plot images will be saved
verbose 						| false,true			| true			| output stats to console
report_freq 					| > 0					| 1				| report results every frame to console
**population**					|						|				|
pop_size 						| > 0					| 2000			| number of agents
mean_age 						| > 0					| 45			| mean age of agents
max_age 						| > 0					| 105			| max age of agents
age_dependent_risk 				| false,true			| true			| whether risk increases with age
risk_age 						| > 0					| 55			| age where mortality risk starts increasing
critical_age 					| > 0					| 75			| age at and beyond which mortality risk reaches maximum
critical_mortality_chance 		| > 0					| 0.1			| maximum mortality risk for older age
risk_increase 					| quadratic, linear		| quadratic 	| how the risk increases with age
**movement**					|						|				|
social_distance_factor 			| >= 0					| 0.0			| how strongly agents repel each other during social distancing
max_speed 						| > 0					| 1.0			| maximum speed of population
dt 								| > 0					| 0.01			| units travelled every timestep at max_speed
thresh_type 					| infected, hospitalized| infected		| type of criterion for turning on interventions
testing_threshold_on 			| >= 0					| 0				| when to start testing (number of infected by default)
social_distance_threshold_on	| >= 0					| 0				| when to start testing (number of infected by default, 0 means start with simulation)
social_distance_threshold_on 	| >= 0					| 0				| when to start distancing (number of infected by default, 0 means start with simulation)
social_distance_threshold_off 	| >= 0					| 0				| when to stop distancing (number of infected by default, 0 means never)
social_distance_violation 		| >= 0					| 0				| number of people that do not follow social distancing
wall_buffer 					| > 0					| 0.001			| wall repulsion zone
bounce_buffer 					| > 0					| 0.0005		| maximum overshoot outside wall
wander_factor_dest 				| > 0					| 1.5			| when people have an active destination, the wander range defines the area surrounding the destination they will wander upon arriving
**infection** 					|						|				|
infection_range 				| > 0					| 0.01			| range surrounding sick patient that infections can take place
infection_shape 				| radial, square		| radial		| shape of infection zone surrounding sick patient that infections can take place
infection_chance 				| > 0					| 0.03			| chance that an infection spreads to nearby healthy people each tick
recovery_duration				| > 0, > 0 				| 200, 500 		| how many ticks it may take to recover from the illness
mortality_chance				| > 0 					| 0.02 			| global baseline chance of dying from the disease
incubation_period				| > 0 					| 0 			| number of frames the individual spreads disease unknowingly
patient_Z_loc					| random, central 		| 0 			| randomly choose patient Z from agents in the center or anywhere at random
**healthcare** 					|						|				|
healthcare_capacity				| > 0 			 		| 300 			| capacity of the healthcare system
treatment_factor				| > 0 			 		| 0.5			| when in treatment, affect risk by this factor
no_treatment_factor				| > 0 			 		| 3				| risk increase factor to use if healthcare system is full
treatment_dependent_risk		| false,true			| true			| whether risk is affected by treatment
**quarantine** 	 				|						|				|
isolation_bounds				| > 0, > 0 				| 0.02, 0.02, 	| lower x and y bounds of the "quarantine" environment
isolation_bounds				| > 0, > 0 				| 0.1, 0.98 	| upper x and y bounds of the "quarantine" environment
number_of_tests					| > 0			 		| 10		 	| number of random tests per day
**random generator**			|						|				|
constant_seed					| false,true			| false			| whether user specifies a constant seed file or not
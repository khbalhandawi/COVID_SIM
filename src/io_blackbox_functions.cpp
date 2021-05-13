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
 \file   io_blackbox_functions.cpp
 \brief  Blackbox functons (implementation)
 \author Khalil Al Handawi
 \date   2010-08-25
 \see    io_blackbox_functions.h
 */


#include "io_blackbox_functions.h"

#include <map>
#include <iostream>
#include <sstream>
#include <iterator>

/*-----------------------------------------------------------*/
/*              Post process simulation results              */
/*-----------------------------------------------------------*/
std::vector<double> COVID_SIM::processInput(int i, simulation *sim, std::ofstream *file)
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
	std::vector<double> matrix_opt; // unstripped matrix
	matrix_opt.push_back(infected);
	matrix_opt.push_back(fatalities);
	matrix_opt.push_back(mean_distance);
	matrix_opt.push_back(mean_GC);

	if (file != nullptr) {
		std::vector<double> matrix_out; // unstripped matrix
		matrix_out.push_back(i);
		matrix_out.push_back(SD);
		matrix_out.push_back(SD_thresh);
		matrix_out.push_back(E);
		matrix_out.push_back(T);
		matrix_out.push_back(infected);
		matrix_out.push_back(fatalities);
		matrix_out.push_back(mean_distance);
		matrix_out.push_back(mean_GC);
		matrix_out.push_back(f);

		// Convert int to ostring stream
		std::ostringstream oss;
		oss.precision(11);
		if (!matrix_out.empty())
		{
			// Convert all but the last element to avoid a trailing ","
			std::copy(matrix_out.begin(), matrix_out.end() - 1,
				std::ostream_iterator<double>(oss, ","));

			// Now add the last element with no delimiter
			oss << matrix_out.back();
		}

		file->precision(11);
		// Write ostring stream to file
		if (file->is_open())
		{
			(*file) << oss.str() << '\n';
		}
	}

	return matrix_opt;

}

/*-----------------------------------------------------------*/
/*                    Load configuration                     */
/*-----------------------------------------------------------*/
void COVID_SIM::load_config(Configuration *config, const char *config_file)
{
	std::map<std::string, std::string Configuration::*> mapper;
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
	mapper["write_bb_output"] = &Configuration::write_bb_output_in;
	mapper["social_distance_factor"] = &Configuration::social_distance_factor_in;
	mapper["social_distance_violation"] = &Configuration::social_distance_violation_in;
	mapper["healthcare_capacity"] = &Configuration::healthcare_capacity_in;
	mapper["number_of_tests"] = &Configuration::number_of_tests_in;
	mapper["self_isolate"] = &Configuration::self_isolate_in;
	mapper["wander_factor_dest"] = &Configuration::wander_factor_dest_in;
	mapper["isolation_bounds"] = &Configuration::isolation_bounds_in;
	mapper["self_isolate_proportion"] = &Configuration::self_isolate_proportion_in;
	mapper["xbounds"] = &Configuration::xbounds_in;
	mapper["ybounds"] = &Configuration::ybounds_in;
	mapper["x_plot"] = &Configuration::x_plot_in;
	mapper["y_plot"] = &Configuration::y_plot_in;
	mapper["traveling_infects"] = &Configuration::traveling_infects_in;
	mapper["save_pop"] = &Configuration::save_pop_in;
	mapper["save_ground_covered"] = &Configuration::save_ground_covered_in;
	mapper["save_pop_freq"] = &Configuration::save_pop_freq_in;
	mapper["infection_shape"] = &Configuration::infection_shape_in;
	mapper["track_R0"] = &Configuration::track_R0_in;
	mapper["update_R0_every_n_frame"] = &Configuration::update_R0_every_n_frame_in;
	mapper["testing_threshold_on"] = &Configuration::testing_threshold_on_in;
	mapper["platform"] = &Configuration::platform_in;
	mapper["wall_buffer"] = &Configuration::wall_buffer_in;
	mapper["bounce_buffer"] = &Configuration::bounce_buffer_in;

	std::ifstream file(config_file); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
	std::string line, value, value_str;

	while (file.good()) {
		getline(file, line, '\n');
		std::istringstream is_line(line);
		std::vector<std::string> key_value_pair;

		while (getline(is_line, value, '=')) {
			key_value_pair.push_back(value.c_str());
		}
		config->*(mapper[key_value_pair[0]]) = key_value_pair[1];
	}
}
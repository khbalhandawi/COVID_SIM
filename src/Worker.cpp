#include "Worker.h"
#include "Population_trackers.h"

#include <QThread>
#include <QCoreApplication>
#include <QVector>
#include <iostream>
#include <fstream>

/*-----------------------------------------------------------*/
/*                        Constructor                        */
/*-----------------------------------------------------------*/
Worker::Worker(COVID_SIM::simulation *sim_in)
{

	sim = sim_in;

	/* run simulation */
	//save grid_coords if required
	if (sim->Config.save_ground_covered) {
		COVID_SIM::save_grid_coords(sim->pop_tracker.grid_coords, sim->Config.save_pop_folder);
	}

	i = 0;

	COVID_SIM::ArrayXXb cond_init(sim->Config.pop_size, 2);
	cond = cond_init;

	emit worker_initialized();

}

/*-----------------------------------------------------------*/
/*             Extract time step results for UI              */
/*-----------------------------------------------------------*/
void Worker::export_results()
{

	// Initialize Qt Vectors
	QVector<double> susceptible_x, susceptible_y,
		infected_x, infected_y,
		recovered_x, recovered_y,
		fatalities_x, fatalities_y,
		tracked_x, tracked_y,
		x_lower, y_lower,
		x_upper, y_upper;

	/*--------------------------------------------------*/
	// plot population segments
	Eigen::ArrayXXd susceptible = sim->population(COVID_SIM::select_rows(sim->population.col(6) == 0), { 1,2 }).cast<double>();
	Eigen::ArrayXXd infected = sim->population(COVID_SIM::select_rows(sim->population.col(6) == 1), { 1,2 }).cast<double>();
	Eigen::ArrayXXd recovered = sim->population(COVID_SIM::select_rows(sim->population.col(6) == 2), { 1,2 }).cast<double>();
	Eigen::ArrayXXd fatalities = sim->population(COVID_SIM::select_rows(sim->population.col(6) == 3), { 1,2 }).cast<double>();

	if (susceptible.rows() > 0) { // to avoid assertion errors
		susceptible_x.resize(susceptible.rows()); Eigen::Map<Eigen::ArrayXd>(&susceptible_x[0], susceptible.rows(), 1) = susceptible.col(0);
		susceptible_y.resize(susceptible.rows()); Eigen::Map<Eigen::ArrayXd>(&susceptible_y[0], susceptible.rows(), 1) = susceptible.col(1);
	}

	if (infected.rows() > 0) { // to avoid assertion errors
		infected_x.resize(infected.rows()); Eigen::Map<Eigen::ArrayXd>(&infected_x[0], infected.rows(), 1) = infected.col(0);
		infected_y.resize(infected.rows()); Eigen::Map<Eigen::ArrayXd>(&infected_y[0], infected.rows(), 1) = infected.col(1);
	}

	if (recovered.rows() > 0) { // to avoid assertion errors
		recovered_x.resize(recovered.rows()); Eigen::Map<Eigen::ArrayXd>(&recovered_x[0], recovered.rows(), 1) = recovered.col(0);
		recovered_y.resize(recovered.rows()); Eigen::Map<Eigen::ArrayXd>(&recovered_y[0], recovered.rows(), 1) = recovered.col(1);
	}

	if (fatalities.rows() > 0) { // to avoid assertion errors
		fatalities_x.resize(fatalities.rows()); Eigen::Map<Eigen::ArrayXd>(&fatalities_x[0], fatalities.rows(), 1) = fatalities.col(0);
		fatalities_y.resize(fatalities.rows()); Eigen::Map<Eigen::ArrayXd>(&fatalities_y[0], fatalities.rows(), 1) = fatalities.col(1);
	}

	float R0 = sim->pop_tracker.mean_R0.back();

	// Trace path of random individual
	if (sim->Config.trace_path) {

		tracked_x.resize(1); Eigen::Map<Eigen::ArrayXd>(&tracked_x[0], 1, 1) = sim->population.block<1, 1>(0, 1).cast<double>();
		tracked_y.resize(1); Eigen::Map<Eigen::ArrayXd>(&tracked_y[0], 1, 1) = sim->population.block<1, 1>(0, 2).cast<double>();

		if ((sim->pop_tracker.ground_covered.row(0) == 1).any()) {
			std::vector<int> rows = COVID_SIM::select_rows(sim->pop_tracker.ground_covered.row(0).transpose() == 1);
			Eigen::ArrayXXf grid = sim->pop_tracker.grid_coords(rows, Eigen::all);

			if (grid.rows() > 0) { // to avoid assertion errors
				x_lower.resize(grid.rows()); Eigen::Map<Eigen::ArrayXd>(&x_lower[0], grid.rows(), 1) = grid.col(0).cast<double>();
				y_lower.resize(grid.rows()); Eigen::Map<Eigen::ArrayXd>(&y_lower[0], grid.rows(), 1) = grid.col(1).cast<double>();
				x_upper.resize(grid.rows()); Eigen::Map<Eigen::ArrayXd>(&x_upper[0], grid.rows(), 1) = grid.col(2).cast<double>();
				y_upper.resize(grid.rows()); Eigen::Map<Eigen::ArrayXd>(&y_upper[0], grid.rows(), 1) = grid.col(3).cast<double>();
			}

		}

	}

	// update mainwindow using new data
	emit resultReady(susceptible_x, susceptible_y,
		infected_x, infected_y,
		recovered_x, recovered_y,
		fatalities_x, fatalities_y,
		tracked_x, tracked_y,
		sim->frame, R0, sim->computation_time,
		x_lower, y_lower, x_upper, y_upper); // emit signal
}

/*-----------------------------------------------------------*/
/*              Post process simulation results              */
/*-----------------------------------------------------------*/
void Worker::processOutputs()
{
	if (sim->Config.save_data) {
		save_data(sim->population, sim->pop_tracker, sim->Config, (sim->frame - 1), sim->Config.save_pop_folder);
	}

	// report outcomes
	if (sim->Config.verbose) {

		cond << (sim->population.col(6) == 1), (sim->population.col(6) == 4);

		Eigen::ArrayXXf pop_susceptible = sim->population(COVID_SIM::select_rows(sim->population.col(6) == 0), Eigen::all); // healthy individuals
		Eigen::ArrayXXf pop_recovered = sim->population(COVID_SIM::select_rows(sim->population.col(6) == 2), Eigen::all); // recovered individuals
		Eigen::ArrayXXf pop_fatality = sim->population(COVID_SIM::select_rows(sim->population.col(6) == 3), Eigen::all); // dead individuals
		Eigen::ArrayXXf pop_asymptomatic = sim->population(COVID_SIM::select_rows(sim->population.col(6) == 4), Eigen::all); // asymptomatic individuals
		Eigen::ArrayXXf pop_infectious = sim->population(COVID_SIM::select_rows_any(cond), Eigen::all); // infectious individuals


		std::cout << "\n\n" << "-----stopping-----" << std::endl;
		std::cout << "total timesteps taken: " << std::to_string(sim->frame) << std::endl;
		std::cout << "total fatalities: " << std::to_string(pop_fatality.rows()) << std::endl;
		std::cout << "total recovered: " << std::to_string(pop_recovered.rows()) << std::endl;
		std::cout << "total infected: " << std::to_string(sim->pop_infected.rows()) << std::endl;
		std::cout << "total infectious: " << std::to_string(pop_infectious.rows()) << std::endl;
		std::cout << "total unaffected: " << std::to_string(pop_susceptible.rows()) << std::endl;
		if (sim->Config.track_GC) {
			std::cout << "Mean % explored: " << sim->pop_tracker.mean_perentage_covered.back() * 100 << std::endl;
		}
		if (sim->Config.track_R0) {
			std::cout << "Max R0: " << *max_element(sim->pop_tracker.mean_R0.begin(), sim->pop_tracker.mean_R0.end()) << std::endl;
		}
	}

	/*-----------------------------------------------------------*/
	/*                    Log blackbox outputs                   */
	/*-----------------------------------------------------------*/

	COVID_SIM::check_folder("data");
	std::string filename = "matlab_out_Blackbox.log";
	std::string full_filename = "data/" + filename;

	// Output evaluation to file
	std::ofstream output_file;

	if (sim->Config.run_i == 0) {
		output_file.open(sim->Config.log_file, std::ofstream::out);

		output_file.precision(11);
		output_file << "index,SD_factor,threshold,essential_workers,testing_capacity," <<
			"n_infected,n_fatalaties,mean_distance,mean_GC,n_steps" << std::endl;
	}
	else {
		output_file.open(sim->Config.log_file, std::ofstream::app);
		output_file.precision(11);
	}

	int infected, fatalities, SD_thresh, E, T, f;
	double mean_distance, mean_GC, SD;

	SD = sim->Config.social_distance_factor / (1e-6 * sim->Config.force_scaling);
	SD_thresh = sim->Config.social_distance_threshold_on;
	E = sim->Config.social_distance_violation;
	T = sim->Config.number_of_tests;

	infected = *max_element(sim->pop_tracker.infectious.begin(), sim->pop_tracker.infectious.end());
	fatalities = sim->pop_tracker.fatalities.back();
	mean_distance = (sim->pop_tracker.distance_travelled.back() / double(sim->frame)) * 100.0 * 2000;
	mean_GC = (sim->pop_tracker.mean_perentage_covered.back() / double(sim->frame)) * 100.0 * 2000;

	f = sim->frame;

	std::vector<double> matrix_out, matrix_opt; // unstripped matrix
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
	std::ostringstream oss;
	oss.precision(11);
	if (!matrix_out.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		copy(matrix_out.begin(), matrix_out.end() - 1,
			std::ostream_iterator<double>(oss, ","));

		// Now add the last element with no delimiter
		oss << matrix_out.back();
	}

	output_file.precision(11);
	// Write ostring stream to file
	if (output_file.is_open())
	{
		(output_file) << oss.str() << '\n';
	}

	output_file.close();

	double obj_1 = -mean_GC;
	double obj_2 = fatalities;
	double c1 = infected - sim->Config.healthcare_capacity;

	if (sim->Config.write_bb_output) {
		std::ofstream output_file_opt(full_filename);
		output_file_opt.precision(10); // number of decimal places to output
		output_file_opt << obj_1 << " " << obj_2 << " " << c1 << std::endl;
		output_file_opt.close();
		std::cout << "obj_1: " << obj_1 << " obj_2: " << obj_2 << " c1: " << c1 << std::endl;
	}

}

void Worker::doWork()
{
	
	/* ... here is the expensive or blocking operation ... */

	if (i < sim->Config.simulation_steps) {

		try
		{
			sim->tstep(); // code that could cause exception
			export_results();
			emit time_step_finished();
		}
		catch (const std::exception &exc)
		{
			// catch anything thrown within try block that derives from std::exception
			std::cerr << exc.what();
			emit simulation_finished();
		}

		// check whether to end if no infectious persons remain.
		// check if frame is above some threshold to prevent early breaking when simulation
		// starts initially with no infections.
		if ((sim->Config.endif_no_infections) && (sim->frame >= 300)) {

			cond << (sim->population.col(6) == 1), (sim->population.col(6) == 4);
			if (sim->population(COVID_SIM::select_rows_any(cond), Eigen::all).rows() == 0) {
				i = sim->Config.simulation_steps;
			}
		}
		i += 1;
	}
	else {
		// i dont know why but worker thread not initializing properly
		if (i == sim->Config.simulation_steps) 	processOutputs(); 
		emit simulation_finished();
	}

}

void Worker::setICValue(int IC_new)
{
	// Set simulation infection_chance from slider
	sim->Config.infection_chance = ((sim->Config.IC_max - sim->Config.IC_min) * (IC_new / 99.0)) + sim->Config.IC_min;
}

void Worker::setSDValue(int SD_new)
{
	// Set simulation SD_factor from slider
	sim->Config.social_distance_factor = 1e-6 * (((sim->Config.SD_max - sim->Config.SD_min) * (SD_new / 99.0)) + sim->Config.SD_min) * sim->Config.force_scaling;
}

void Worker::setTCValue(int TC_new)
{
	// Set simulation number_of_tests from slider
	sim->Config.number_of_tests = int(((sim->Config.TC_max - sim->Config.TC_min) * (TC_new / 99.0)) + sim->Config.TC_min);
}

Worker::~Worker()
{
}

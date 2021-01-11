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
 \file   Population_trackers.cpp
 \brief  Initialization and tracking of population parameters (implementation)
 \author Khalil Al Handawi
 \date   2010-08-25
 \see    Population_trackers.h
 */

#include "Population_trackers.h"
#include "Convert.h"
#ifdef GPU_ACC
#include "CUDA_functions.h"
#endif // GPU_ACC

/*-----------------------------------------------------------*/
/*                       Constructor                         */
/*-----------------------------------------------------------*/
Population_trackers::Population_trackers(Configuration Config_init)
{
	susceptible = { };
	infectious = { };
	recovered = { };
	fatalities = { };
	Config = Config_init;
	distance_travelled = { };
	total_distance = Eigen::ArrayXf::Zero(Config_init.pop_size, 1); // distance travelled by individuals
	mean_perentage_covered = { };
	mean_R0 = { };
	perentage_covered = Eigen::ArrayXf::Zero(Config_init.pop_size, 1); // portion of world covered by individuals
	// PLACEHOLDER - whether recovered individual can be reinfected
	reinfect = false;
}

/*-----------------------------------------------------------*/
/*                      Update counts                        */
/*-----------------------------------------------------------*/
void Population_trackers::update_counts(Eigen::ArrayXXf population, int frame)
{

	/*docstring
	*/
	int pop_size = int(population.rows());
	
	int n_infected = int((population.col(6) == 1).count());
	int n_recovered = int((population.col(6) == 2).count());
	int n_fatalities = int((population.col(6) == 3).count());

	infectious.push_back(n_infected);
	recovered.push_back(n_recovered);
	fatalities.push_back(n_fatalities);

	Eigen::ArrayXXf ground_covered_zero = Eigen::ArrayXXf::Zero(pop_size, int(pow((Config.n_gridpoints - 1), 2)) );

	// Total distance travelled
	if (Config.track_position) {
		Eigen::ArrayXXf speed_vector = population(select_rows(population.col(11) == 0), { 3,4 }); // speed of individuals within world
		Eigen::ArrayXf distance_individuals = speed_vector.rowwise().norm() * Config.dt; // current distance travelled

		total_distance(select_rows(population.col(11) == 0), all) += distance_individuals; // cumulative distance travelled
		distance_travelled.insert(distance_travelled.end(), total_distance.mean()); // mean cumulative distance
	}
	else {
		distance_travelled.push_back(0.0); // mean cumulative distance
	}

	// Compute and track ground covered
	if (Config.track_GC) {
		if (frame % Config.update_every_n_frame == 0) {

			// Track ground covered
			int n_inside_world = (population.col(11) == 0).count();
			Eigen::ArrayXXf position_vector = population(select_rows(population.col(11) == 0), { 1,2 }); // position of individuals within world
			Eigen::ArrayXXf GC_matrix = ground_covered(select_rows(population.col(11) == 0), all);
#ifdef GPU_ACC
			Eigen::ArrayXf p(n_inside_world); // Initialize percentage arrays
			tracker_gpu(&GC_matrix, &p, position_vector.col(0), position_vector.col(1), n_inside_world, Config.n_gridpoints - 1, 1024);

			ground_covered(select_rows(population.col(11) == 0), all) = GC_matrix;

			// count number of non-zeros rowwise
			perentage_covered(select_rows(population.col(11) == 0)) = p / (Config.n_gridpoints * Config.n_gridpoints);
			mean_perentage_covered.push_back(perentage_covered.mean()); // mean ground covered
#else
			// 1D
			Eigen::ArrayXf pos_vector_x = position_vector.col(0);
			Eigen::ArrayXf pos_vector_y = position_vector.col(1);

			Eigen::ArrayXXf g1 = grid_coords.col(0).replicate(1, n_inside_world);
			Eigen::ArrayXXf g2 = grid_coords.col(1).replicate(1, n_inside_world);
			Eigen::ArrayXXf g3 = grid_coords.col(2).replicate(1, n_inside_world);
			Eigen::ArrayXXf g4 = grid_coords.col(3).replicate(1, n_inside_world);

			Eigen::ArrayXXf l_x = -1 * (g1.rowwise() - pos_vector_x.transpose()).transpose();
			Eigen::ArrayXXf l_y = -1 * (g2.rowwise() - pos_vector_y.transpose()).transpose();
			Eigen::ArrayXXf u_x = (g3.rowwise() - pos_vector_x.transpose()).transpose();
			Eigen::ArrayXXf u_y = (g4.rowwise() - pos_vector_y.transpose()).transpose();

			ArrayXXb conds = (l_x > 0) && (u_x > 0) && (l_y > 0) && (u_y > 0);

			GC_matrix += conds.cast<float>();
			ground_covered(select_rows(population.col(11) == 0), all) = GC_matrix;

			// count number of non-zeros rowwise
			perentage_covered = (ground_covered != 0).rowwise().count().cast<float>() / grid_coords.rows();
			mean_perentage_covered.push_back(perentage_covered.mean()); // mean ground covered
#endif
		}
	}
	else {
		mean_perentage_covered.push_back(0.0); // mean ground covered
	}

	// Compute and track R0
	if (Config.track_R0) {
		if (frame % Config.update_R0_every_n_frame == 0) {
			double mean_infection_time = (Config.recovery_duration[0] + Config.recovery_duration[1]) / 2; // how many ticks it may take to recover from the illness

			vector<int> rows_IRF = select_rows(population.col(6) != 0);
			// If there are non-healthy people present
			if (rows_IRF.size() > 0) {
				Eigen::ArrayXXf pop_IRF = population(rows_IRF, Eigen::all);
				Eigen::ArrayXf prop = (frame - pop_IRF.col(8)) / (pop_IRF.col(19) != 0.0).select(mean_infection_time, frame - pop_IRF.col(19));
				
				vector<int> R0_rows = select_rows(prop >= 0.1);
				// if prop values above threshold for computing R0
				if (R0_rows.size() > 0) {
					Eigen::ArrayXf R0_values = pop_IRF(R0_rows, {20} ) / prop(R0_rows);
					mean_R0.push_back(R0_values.mean()); // basic reproductive number
				} else {
					mean_R0.push_back(0.0); // basic reproductive number
				}
			} else {
				mean_R0.push_back(0.0); // basic reproductive number
			}
		}
	} else {
		mean_R0.push_back(0.0); // basic reproductive number
	}

	// Mark recovered individuals as susceptable if reinfection enables
	if (reinfect) {
		susceptible.push_back( pop_size - (infectious.back() + fatalities.back()) );
	}
	else {
		susceptible.push_back( pop_size - (infectious.back() + recovered.back() + fatalities.back()) );
	}

}

/*-----------------------------------------------------------*/
/*                        Destructor                         */
/*-----------------------------------------------------------*/
Population_trackers::~Population_trackers()
{
}

/*-----------------------------------------------------------*/
/*                  initialize population                    */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf initialize_population(Configuration Config, RandomDevice *my_rand, int mean_age, int max_age,
	vector<double> xbounds, vector<double> ybounds)
{
	/*initialized the population for the simulation

	the population matrix for this simulation has the following columns :

	0 : unique ID
	1 : current x coordinate
	2 : current y coordinate
	3 : current heading in x direction
	4 : current heading in y direction
	5 : current speed
	6 : current state(0 = healthy, 1 = sick, 2 = immune, 3 = dead, 4 = immune but infectious)
	7 : age
	8 : infected_since(frame the person got infected)
	9 : recovery vector(used in determining when someone recovers or dies)
	10 : in treatment
	11 : active destination(0 = random wander, 1, .. = destination matrix index)
	12 : at destination : whether arrived at destination(0 = traveling, 1 = arrived)
	13 : wander_range_x : wander ranges on x axis for those who are confined to a location
	14 : wander_range_y : wander ranges on y axis for those who are confined to a location
	15 : total force x
	16 : total force y
	17 : violator(0: compliant 1 : violator)
	18 : flagged for testing(0: no 1 : yes)
	19 : removed_since (frame the person got recovered or dead)
	20 : number of people transimitted to

	Keyword arguments
	---------------- -
	pop_size : int
	the size of the population

	mean_age : int
	the mean age of the population.Age affects mortality chances

	max_age : int
	the max age of the population

	xbounds : 2d array
	lower and upper bounds of x axis

	ybounds : 2d array
	lower and upper bounds of y axis
	*/
	float epsilon = 1e-15;
	// initialize population matrix
	Eigen::ArrayXXf population = Eigen::ArrayXXf::Zero(Config.pop_size, 21);
	// initalize unique IDs
	population.col(0) = Eigen::ArrayXf::LinSpaced(Config.pop_size, 0, Config.pop_size - 1);

	// initialize random coordinates
	population.block(0,1,Config.pop_size,2) = my_rand->uniform_dist(xbounds[0] + 0.05, xbounds[1] - 0.05, Config.pop_size, 2);

	// initialize random speeds - 0.25 to 0.25
	Eigen::ArrayXXf vect_un = my_rand->uniform_dist(-1, 1, Config.pop_size, 2);

	Eigen::ArrayXXf speed_vector = vect_un.array().colwise() / ( vect_un.rowwise().norm() + epsilon );
	population(Eigen::all, { 3,4 }) = Config.max_speed * speed_vector;

	// initalize ages
	double std_age = (max_age - mean_age) / 3;
	population.col(7) = my_rand->normal_dist(mean_age, std_age, Config.pop_size, 1).round();
	population.col(7) = population.col(7).max(0).min(max_age); // clip those younger than 0 years

	// build recovery_vector
	population.col(9) = my_rand->normal_dist(0.5, 0.5 / 3, Config.pop_size, 1);

	//// Randomly place people outside (comment when not debugging)
	//population.col(11) = my_rand->Random_choice_prob(Config.pop_size, 0.5);
	
	// Randomly select social distancing violators
	Eigen::VectorXi Choices = my_rand->Random_choice(population.col(0), Config.social_distance_violation);
	population(Choices, { 17 }) = 1;

	return population;
}

/*-----------------------------------------------------------*/
/*                 initialize destinations                   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf initialize_destination_matrix(int pop_size, int total_destinations)
{
	/*intializes the destination matrix

	function that initializes the destination matrix used to
	define individual location and roam zones for population members

	Keyword arguments
	---------------- -
	pop_size : int
	the size of the population

	total_destinations : int
	the number of destinations to maintain in the matrix.Set to more than
	one if for example people can go to work, supermarket, home, etc.
	*/

	Eigen::ArrayXXf destinations = Eigen::ArrayXXf::Zero(pop_size, total_destinations * 2);

	return destinations;
}


/*-----------------------------------------------------------*/
/*                initialize ground covered                  */
/*-----------------------------------------------------------*/
void initialize_ground_covered_matrix(Eigen::ArrayXXf &grid_coords, Eigen::ArrayXXf &ground_covered, int pop_size, int n_gridpoints, vector<double> xbounds,
	vector<double> ybounds)
{
	/*intializes the destination matrix

	function that initializes the destination matrix used to
	define individual location and roam zones for population members

	Keyword arguments
	---------------- -
	pop_size : int
	the size of the population

	n_gridpoints : int
	resolution of the grid dimensions in 1D

	xbounds : 2d array
	lower and upper bounds of x axis

	ybounds : 2d array
	lower and upper bounds of y axis
	*/

	Eigen::ArrayXf x = Eigen::ArrayXf::LinSpaced(n_gridpoints, xbounds[0], xbounds[1]);
	Eigen::ArrayXf y = Eigen::ArrayXf::LinSpaced(n_gridpoints, ybounds[0], ybounds[1]);

	// create list of grid points and their bounding boxes
	Eigen::ArrayXf grid_coords_xlb = x(seq(0, last - 1)).replicate(n_gridpoints - 1, 1);
	Eigen::ArrayXf grid_coords_ylb = repeat(y(seq(0, last - 1)), n_gridpoints - 1);
	Eigen::ArrayXf grid_coords_xub = x(seq(1, last)).replicate(n_gridpoints - 1, 1);
	Eigen::ArrayXf grid_coords_yub = repeat(y(seq(1, last)), n_gridpoints - 1);

	grid_coords.resize(grid_coords_xlb.rows(), grid_coords_xlb.cols()*4);
	grid_coords << grid_coords_xlb, grid_coords_ylb, grid_coords_xub, grid_coords_yub;

	ground_covered = Eigen::ArrayXXf::Zero(pop_size, pow((n_gridpoints - 1),2) );

	// return { grid_coords, ground_covered };
}

/*-----------------------------------------------------------*/
/*                   destination bounds                      */
/*-----------------------------------------------------------*/
void set_destination_bounds(Eigen::ArrayXXf &population, Eigen::ArrayXXf &destinations, double xmin, double ymin,
	double xmax, double ymax, RandomDevice *my_rand, int dest_no, bool teleport)
{
	/*teleports all persons within limits

	Function that takes the population and coordinates,
	teleports everyone there, sets destination active and
	destination as reached

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	destinations : ndarray
	the array containing all the destination information

	xmin, ymin, xmax, ymax : int or float
	define the bounds on both axes where the individual can roam within
	after reaching the defined area

	dest_no : int
	the destination number to set as active(if more than one)

	teleport : bool
	whether to instantly teleport individuals to the defined locations
	*/

	// teleport
	if (teleport) {
		population.col(1) = my_rand->uniform_dist(xmin, xmax, population.rows(), 1);
		population.col(2) = my_rand->uniform_dist(ymin, ymax, population.rows(), 1);
	}

	// get parameters
	double x_center, y_center, x_wander, y_wander;
	vector<double> outputs = get_motion_parameters(xmin, ymin, xmax, ymax);

	x_center = outputs[0];
	y_center = outputs[1];
	x_wander = outputs[2];
	y_wander = outputs[3];

	// set destination centers
	destinations.col( (dest_no - 1) * 2 ) = x_center;
	destinations.col( ( (dest_no - 1) * 2 ) + 1 ) = y_center;

	// set wander bounds
	population.col(13) = x_wander;
	population.col(14) = y_wander;

	population.col(11) = dest_no; //set destination active
	population.col(12) = 1; // set destination reached

}

/*-----------------------------------------------------------*/
/*                   save population data                    */
/*-----------------------------------------------------------*/
void save_data(Eigen::ArrayXXf population, Population_trackers pop_tracker, Configuration Config, int frame, string folder)
{
	/*dumps simulation data to disk

	Function that dumps the simulation data to specific files on the disk.
	Saves final state of the population matrix, the array of infected over time,
	and the array of fatalities over time

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	infected : list or ndarray
	the array containing data of infections over time

	fatalities : list or ndarray
	the array containing data of fatalities over time
	*/

	check_folder(folder);

	vector<int> *s = &pop_tracker.susceptible;
	vector<int> *i = &pop_tracker.infectious;
	vector<int> *r = &pop_tracker.recovered;
	vector<int> *f = &pop_tracker.fatalities;
	vector<int> t = sequence(0, (frame + 1), 1);

	vector<double> *d = &pop_tracker.distance_travelled;
	vector<double> *GC = &pop_tracker.mean_perentage_covered;
	vector<double> *R0 = &pop_tracker.mean_R0;

	Eigen::Map<Eigen::MatrixXi> susceptible(s->data(), s->size(), 1);
	Eigen::Map<Eigen::MatrixXi> infectious(i->data(), i->size(), 1);
	Eigen::Map<Eigen::MatrixXi> recovered(r->data(), r->size(), 1);
	Eigen::Map<Eigen::MatrixXi> fatalities(f->data(), f->size(), 1);
	Eigen::Map<Eigen::MatrixXi> time(t.data(), t.size(), 1);

	Eigen::MatrixXi SIRF_data(s->size(),5);
	SIRF_data << time, susceptible, infectious, recovered, fatalities;

	IOFile::writeTofile(SIRF_data.cast<double>(), folder + "\\SIRF_data.bin");

	if (Config.track_position) {
		Eigen::Map<Eigen::MatrixXd> distance_travelled(d->data(), d->size(), 1);

		vector<int> t = sequence(0, (frame + 1), 1);
		Eigen::Map<Eigen::MatrixXi> time(t.data(), t.size(), 1);

		Eigen::MatrixXd time_series(distance_travelled.rows(),2);
		time_series << time.cast<double>(), distance_travelled;

		IOFile::writeTofile(time_series, folder + "\\dist_data.bin");
	}
	if (Config.track_GC) {
		Eigen::Map<Eigen::MatrixXd> mean_perentage_covered(GC->data(), GC->size(), 1);

		vector<int> t = sequence(0, (frame + 1), Config.update_every_n_frame);
		Eigen::Map<Eigen::MatrixXi> time(t.data(), t.size(), 1);

		Eigen::MatrixXd time_series(mean_perentage_covered.rows(),2);
		time_series << time.cast<double>(), mean_perentage_covered;

		IOFile::writeTofile(time_series, folder + "\\mean_GC_data.bin");
	}
	if (Config.track_R0) {
		Eigen::Map<Eigen::MatrixXd> mean_R0(R0->data(), R0->size(), 1);

		vector<int> t = sequence(0, (frame + 1), Config.update_R0_every_n_frame);
		Eigen::Map<Eigen::MatrixXi> time(t.data(), t.size(), 1);

		Eigen::MatrixXd time_series(mean_R0.rows(),2);
		time_series << time.cast<double>(), mean_R0;

		IOFile::writeTofile(time_series, folder + "\\mean_R0_data.bin");
	}

}

/*-----------------------------------------------------------*/
/*         save population data at current time step         */
/*-----------------------------------------------------------*/
void save_population(Eigen::ArrayXXf population, int tstep, string folder)
{
	/*dumps population data at given timestep to disk

	Function that dumps the simulation data to specific files on the disk.
	Saves final state of the population matrix

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	tstep : int
	the timestep that will be saved
	*/
	check_folder(folder);
	IOFile::writeTofile(population.cast<double>(), folder + "\\population_" + to_string(tstep) + ".bin");
}

/*-----------------------------------------------------------*/
/*    save population ground covered at current time step    */
/*-----------------------------------------------------------*/
void save_ground_covered(Eigen::ArrayXXf ground_covered, int tstep, string folder)
{
	/*dumps population data at given timestep to disk

	Function that dumps the population tracking data to specific files on the disk.
	Saves final state of the ground_covered matrix

	Keyword arguments
	---------------- -
	ground_covered : ndarray
	the array containing all the population information

	tstep : int
	the timestep that will be saved
	*/
	check_folder(folder);
	IOFile::writeTofile(ground_covered.cast<double>(), folder + "\\ground_covered_" + to_string(tstep) + ".bin");
}

/*-----------------------------------------------------------*/
/*                   save grid coordinates                   */
/*-----------------------------------------------------------*/
void save_grid_coords(Eigen::ArrayXXf grid_coords, string folder)
{
	/*dumps population data at given timestep to disk

	Function that dumps the tracking grid coordinates to specific files on the disk.
	Saves the grid_coords matrix

	Keyword arguments
	---------------- -
	grid_coords : ndarray
	the array containing all the population information

	*/
	check_folder(folder);
	IOFile::writeTofile(grid_coords.cast<double>(), folder + "\\grid_coords" + ".bin");
}
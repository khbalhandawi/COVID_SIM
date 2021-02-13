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
 \file   motion.cpp
 \brief  Methods related to population mobility and related computations (implementation)
 \author Khalil Al Handawi
 \date   2010-08-25
 \see    motion.h
 */

#include "motion.h"

#include "utilities.h"
#include "RandomDevice.h"
#ifdef GPU_ACC
#include "CUDA_functions.h"
#endif // GPU_ACC

 /*-----------------------------------------------------------*/
 /*                      Update positions                     */
 /*-----------------------------------------------------------*/
void COVID_SIM::update_positions(Eigen::ArrayXXf &population, double dt)
{
	/*update positions of all people

	Uses heading and speed to update all positions for
	the next time step

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	dt : float
	Time increment used for incrementing velocity due to forces
	*/

	// update positions
	// x
	population.col(1) += population.col(3) * dt;
	// y
	population.col(2) += population.col(4) * dt;
}

/*-----------------------------------------------------------*/
/*                      Update velocities                    */
/*-----------------------------------------------------------*/
void COVID_SIM::update_velocities(Eigen::ArrayXXf &population_all, double max_speed, double dt)
{
	/*update positions of all people

	Uses heading and speed to update all positions for
	the next time step

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	max_speed : float
	Maximum speed cap for individuals

	dt : float
	Time increment used for incrementing velocity due to forces
	*/

	// update selectively
	ArrayXXb cond(population_all.rows(), 2);
	Eigen::ArrayXXf population = population_all(select_rows(population_all.col(12) == 1), Eigen::all);
	
	float epsilon = 1e-15;
	// Apply force
	population.col(3) += population.col(15) * dt;
	population.col(4) += population.col(16) * dt;

	// Limit speed
	Eigen::ArrayXf speed = population(Eigen::all, { 3,4 }).rowwise().norm(); // current distance travelled
	population(select_rows(speed > max_speed), { 3,4 }).colwise() *= max_speed / ( speed(select_rows(speed > max_speed)) + epsilon );

	// Limit force
	population(Eigen::all, { 15,16 }) = 0.0;

	// Update velocities
	population_all(select_rows(population_all.col(12) == 1), Eigen::all) = population;

}

/*-----------------------------------------------------------*/
/*                      Update wall forces                   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf COVID_SIM::update_wall_forces(Eigen::ArrayXXf population, Eigen::ArrayXXf xbounds, Eigen::ArrayXXf ybounds,
	double wall_buffer, double bounce_buffer)
{
	/*checks which people are about to go out of bounds and corrects

	Function that calculates wall repulsion forces on individuals that are about to
	go outside of the world boundaries.

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	xbounds, ybounds : list or tuple
	contains the lower and upper bounds of the world[min, max]

	wall_buffer, bounce_buffer : float
	buffer used for wall force calculation and returning
	individuals within bounds
	*/

	int pop_size = population.rows();

	// Avoid walls
	Eigen::ArrayXXf wall_force = Eigen::ArrayXXf::Zero(pop_size, 2);

	float epsilon = 1e-15;

	Eigen::ArrayXf to_lower_x = population.col(1) - xbounds.col(0);
	Eigen::ArrayXf to_lower_y = population.col(2) - ybounds.col(0);

	Eigen::ArrayXf to_upper_x = xbounds.col(1) - population.col(1);
	Eigen::ArrayXf to_upper_y = ybounds.col(1) - population.col(2);

	// Bounce individuals within the world
	ArrayXXb bounce_lo_x(to_lower_x.rows(), 2), bounce_lo_y(to_lower_x.rows(), 2);
	bounce_lo_x << (to_lower_x > -bounce_buffer), (to_lower_x < 0.0); // , (population.col(3) < 0);
	bounce_lo_y << (to_lower_y > -bounce_buffer), (to_lower_y < 0.0); // , (population.col(4) < 0);

	population(select_rows(bounce_lo_x), { 3 }) = population(select_rows(bounce_lo_x), { 3 }).abs();
	population(select_rows(bounce_lo_y), { 4 }) = population(select_rows(bounce_lo_y), { 4 }).abs();

	population(select_rows(bounce_lo_x), { 1 }) = xbounds(select_rows(bounce_lo_x), { 0 }) + bounce_buffer;
	population(select_rows(bounce_lo_y), { 2 }) = ybounds(select_rows(bounce_lo_y), { 0 }) + bounce_buffer;

	ArrayXXb bounce_ur_x(to_lower_x.rows(), 2), bounce_ur_y(to_lower_x.rows(), 2);
	bounce_ur_x << (to_upper_x > -bounce_buffer), (to_upper_x < 0.0); // , (population.col(3) < 0);
	bounce_ur_y << (to_upper_y > -bounce_buffer), (to_upper_y < 0.0); // , (population.col(4) < 0);

	population(select_rows(bounce_ur_x), { 3 }) = -population(select_rows(bounce_ur_x), { 3 }).abs();
	population(select_rows(bounce_ur_y), { 4 }) = -population(select_rows(bounce_ur_y), { 4 }).abs();

	population(select_rows(bounce_ur_x), { 1 }) = xbounds(select_rows(bounce_ur_x), { 1 }) - bounce_buffer;
	population(select_rows(bounce_ur_y), { 2 }) = ybounds(select_rows(bounce_ur_y), { 1 }) - bounce_buffer;

	// Attract outside individuals returning to the world
	ArrayXXb lo_outside_x(to_lower_x.rows(), 1), lo_outside_y(to_lower_y.rows(), 1);
	lo_outside_x << (to_lower_x < 0.0); // , (population.col(3) < 0);
	lo_outside_y << (to_lower_y < 0.0); // , (population.col(4) < 0);

	wall_force(select_rows(lo_outside_x), { 0 }) += 1.0 / (to_lower_x(select_rows(lo_outside_x)).pow(1).abs() + epsilon);
	wall_force(select_rows(lo_outside_y), { 1 }) += 1.0 / (to_lower_y(select_rows(lo_outside_y)).pow(1).abs() + epsilon);
	// wall_force(select_rows(lo_outside_x), { 0 }) -= (1.0 / to_lower_x(select_rows(lo_outside_x)).pow(1)).abs();
	// wall_force(select_rows(lo_outside_y), { 0 }) -= (1.0 / to_lower_x(select_rows(lo_outside_y)).pow(1)).abs();

	ArrayXXb ur_outside_x(to_upper_x.rows(), 1), ur_outside_y(to_upper_y.rows(), 1);
	ur_outside_x << (to_upper_x < 0.0); // , (population.col(3) < 0);
	ur_outside_y << (to_upper_y < 0.0); // , (population.col(4) < 0);

	wall_force(select_rows(ur_outside_x), { 0 }) += 1.0 / (to_lower_x(select_rows(ur_outside_x)).pow(1).abs() + epsilon);
	wall_force(select_rows(ur_outside_y), { 1 }) += 1.0 / (to_lower_y(select_rows(ur_outside_y)).pow(1).abs() + epsilon);
	// wall_force(select_rows(ur_outside_x), { 0 }) -= (1.0 / to_upper_x(select_rows(ur_outside_x)).pow(1)).abs();
	// wall_force(select_rows(ur_outside_y), { 0 }) -= (1.0 / to_upper_y(select_rows(ur_outside_y)).pow(1)).abs();

	// // Repelling force
	// wall_force[:, i] += np.maximum((-1 / wall_buffer * *1 + 1 / to_lower * *1), 0)
	// wall_force[:, i] -= np.maximum((-1 / wall_buffer * *1 + 1 / to_upper * *1), 0)

	// Repelling force
	// wall_force[:, i] += np.maximum((1 / to_lower), 0)
	// wall_force[:, i] -= np.maximum((1 / to_upper), 0)

	ArrayXXb inside_wall_lower_x(to_lower_x.rows(), 2), inside_wall_lower_y(to_lower_y.rows(), 2);
	inside_wall_lower_x << (to_lower_x < wall_buffer), (to_lower_x > -bounce_buffer);
	inside_wall_lower_y << (to_lower_y < wall_buffer), (to_lower_y > -bounce_buffer);

	ArrayXXb inside_wall_upper_x(to_upper_x.rows(), 2), inside_wall_upper_y(to_upper_y.rows(), 2);
	inside_wall_upper_x << (to_upper_x < wall_buffer), (to_lower_x > -bounce_buffer);
	inside_wall_upper_y << (to_upper_y < wall_buffer), (to_lower_y > -bounce_buffer);

	Eigen::ArrayXf tlx_s = to_lower_x(select_rows(inside_wall_lower_x)).abs() + epsilon;
	Eigen::ArrayXf tly_s = to_lower_y(select_rows(inside_wall_lower_y)).abs() + epsilon;
	Eigen::ArrayXf tux_s = to_upper_x(select_rows(inside_wall_upper_x)).abs() + epsilon;
	Eigen::ArrayXf tuy_s = to_upper_y(select_rows(inside_wall_upper_y)).abs() + epsilon;

	wall_force(select_rows(inside_wall_lower_x), { 0 }) += 1.0 / ( tlx_s * tlx_s );
	wall_force(select_rows(inside_wall_lower_y), { 1 }) += 1.0 / ( tly_s * tly_s );

	wall_force(select_rows(inside_wall_upper_x), { 0 }) -= 1.0 / ( tux_s * tux_s );
	wall_force(select_rows(inside_wall_upper_y), { 1 }) -= 1.0 / ( tuy_s * tuy_s );

	population(Eigen::all, { 15,16 }) += wall_force;

	return population;

}

#ifdef GPU_ACC
/*-----------------------------------------------------------*/
/*                   Update repulsive forces                 */
/*-----------------------------------------------------------*/
void COVID_SIM::update_repulsive_forces_cuda(Eigen::ArrayXXf &population_all, double social_distance_factor, CUDA_GPU::Kernels *ABM_cuda)
{
	/*calculated repulsive forces between individuals

	Function that calculates repulsion forces between individuals during social distancing.

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	social_distance_factor : float
	Amplitude of repulsive force used to enforce social distancing
	*/

	// update selectively
	ArrayXXb cond(population_all.rows(), 3);
	cond << (population_all.col(17) == 0), (population_all.col(11) == 0), (population_all.col(12) == 1);
	std::vector<int> rows_cond = select_rows(cond);

	Eigen::ArrayXXf population = population_all(rows_cond, Eigen::all);
	int pop_size = population.rows();

	Eigen::ArrayXf repulsion_force_x(pop_size), repulsion_force_y(pop_size);
	ABM_cuda->pairwise_gpu(population.col(1), population.col(2), social_distance_factor);
	ABM_cuda->get_forces(&repulsion_force_x, &repulsion_force_y);
	
	// Update forces
	population_all.col(15)(rows_cond) += repulsion_force_x;
	population_all.col(16)(rows_cond) += repulsion_force_y;
}

#else
/*-----------------------------------------------------------*/
/*                   Update repulsive forces                 */
/*-----------------------------------------------------------*/
void COVID_SIM::update_repulsive_forces(Eigen::ArrayXXf &population_all, double social_distance_factor, Eigen::ArrayXXf &dist_all, bool compute_dist_all)
{
	/*calculated repulsive forces between individuals

	Function that calculates repulsion forces between individuals during social distancing.

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	social_distance_factor : float
	Amplitude of repulsive force used to enforce social distancing
	*/

	// update selectively
	ArrayXXb cond(population_all.rows(), 3);
	cond << (population_all.col(17) == 0), (population_all.col(11) == 0), (population_all.col(12) == 1);
	std::vector<int> rows_cond = select_rows(cond);

	Eigen::ArrayXXf population = population_all(rows_cond, Eigen::all);
	int pop_size = population.rows();

	Eigen::ArrayXXf dist;

	if (compute_dist_all) {
		// complete pairwise distance for entire population
		dist_all = pairwise_dist(population_all(Eigen::all, { 1,2 }));
		dist = dist_all(rows_cond, rows_cond);
	} else {
		 // complete pairwise distance for non-essential inside world population
		dist = pairwise_dist(population_all(rows_cond, { 1,2 }));
	}

	float epsilon = 1e-15; // to avoid division by zero errors
	// dist += Eigen::MatrixXf::Identity(pop_size, pop_size).array();

	Eigen::ArrayXXf to_point_x = pairwise_diff(population.col(1));
	Eigen::ArrayXXf to_point_y = pairwise_diff(population.col(2));

	Eigen::ArrayXf repulsion_force_x = -social_distance_factor * (to_point_x / ((dist * dist * dist) + epsilon)).rowwise().sum();
	Eigen::ArrayXf repulsion_force_y = -social_distance_factor * (to_point_y / ((dist * dist * dist) + epsilon)).rowwise().sum();

	population.col(15) += repulsion_force_x;
	population.col(16) += repulsion_force_y;

	// Update forces
	population_all(rows_cond, Eigen::all) = population;

}
#endif // GPU_ACC

/*-----------------------------------------------------------*/
/*                    Update gravity forces                  */
/*-----------------------------------------------------------*/
void COVID_SIM::update_gravity_forces(Eigen::ArrayXXf &population, double time, double &last_step_change, RandomDevice *my_rand, double wander_step_size,
	double gravity_strength, double wander_step_duration)
{
	/*updates random perturbation in forces near individuals to cause random motion

	Function that returns geometric parameters of the destination
	that the population members have set.

	Keyword arguments :
	------------------
	population : ndarray
	the array containing all the population information

	time : float
	current simulation time

	last_step_change : float
	last time value at which a random perturbation was introduced

	wander_step_size, gravity_strength, wander_step_duration : float
	proximity of perturbation to individuals,
	strength of attracion to perturbation,
	length of time perturbation is present
	*/

	float epsilon = 1e-15;
	int pop_size = population.rows();

	// Gravity
	if (wander_step_size != 0.0) {
		if ((time - last_step_change) > wander_step_duration)
		{
			Eigen::ArrayXXf vect_un = my_rand->uniform_dist(-1, 1, pop_size, 2);
			Eigen::ArrayXXf vect = vect_un.colwise() / ( vect_un.rowwise().norm().array() + epsilon );

			Eigen::ArrayXXf gravity_well = population(Eigen::all, { 1,2 }) + wander_step_size * vect;
			last_step_change = time;

			Eigen::ArrayXXf to_well = (gravity_well - population(Eigen::all, { 1,2 }));
			Eigen::ArrayXf dist = to_well.rowwise().norm().array();

			population(select_rows(dist != 0), { 15 }) += gravity_strength * to_well(select_rows(dist != 0), { 0 }) / (dist(select_rows(dist != 0)).pow(3) + epsilon);
			population(select_rows(dist != 0), { 16 }) += gravity_strength * to_well(select_rows(dist != 0), { 1 }) / (dist(select_rows(dist != 0)).pow(3) + epsilon);
		}

	}

}
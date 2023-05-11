/*---------------------------------------------------------------------------------*/
/*  COVID_SIM_GPU - Agent-based model of a pandemic in a population -         	   */
/*                                                                                 */
/*  COVID_SIM_GPU - version 1.0.0 has been created by                              */
/*                 Khalil Al Handawi           - McGill University                 */
/*                                                                                 */
/*  The copyright of COVID_SIM_GPU is owned by                                     */
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
 \file   path_planning.cpp
 \brief  Methods related to goal-directed traveling behaviour and path planning (implementation)
 \author Khalil Al Handawi
 \date   2010-08-25
 \see    path_planning.h
 */

#include "path_planning.h"
#include "utilities.h"
#include "motion.h"

/*-----------------------------------------------------------*/
/*                  Send patient to location                 */
/*-----------------------------------------------------------*/
void COVID_SIM::go_to_location(Eigen::VectorXi Choices, Eigen::ArrayXXf &patients, Eigen::ArrayXXf &destinations, int dest_no)
{
	/*sends patient to defined location

	Function that takes a patient an destination, and sets the location
	as active for that patient.

	Keyword arguments
	---------------- -
	Choices : int array
	the array containing ids of people to send to destination

	patients : ndarray
	the array containing all the population information

	destinations : ndarray
	the array containing all destinations information

	dest_no : int
	the location number, used as index for destinations array if multiple possible
	destinations are defined`.


	TODO : vectorize

	*/

	for (int i = 0; i < Choices.size(); i++) {
		int current_dest = patients(Choices[i], 11);

		destinations(Choices[i], (current_dest) * 4) = patients(Choices[i], 1); // save last known location at current destination
		destinations(Choices[i], ((current_dest) * 4) + 1) = patients(Choices[i], 2); // save last known location at current destination

	}

	patients(Choices, { 11 }) = dest_no; // set destination active
	patients(Choices, { 12 }) = 0; // set patient as travelling
}

/*-----------------------------------------------------------*/
/*                Set population destination                 */
/*-----------------------------------------------------------*/
void COVID_SIM::set_destination(Eigen::ArrayXXf &population, Eigen::ArrayXXf destinations, double travel_speed)
{
	/*sets destination of population

	Sets the destination of population if destination marker is not 0.
	Updates headings and speeds as well.

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	destinations : ndarray
	the array containing all destinations information
	*/
	float epsilon = 1e-15;
	// how many destinations are active
	Eigen::ArrayXf dests = population.col(11);
	std::vector<float> active_dests(dests.rows()); Eigen::Map<Eigen::ArrayXf>(&active_dests[0], dests.rows(), 1) = dests;
	unique_elements(active_dests);

	// set destination
	for (int d : active_dests) {

		// pick x and y columns for given d
		Eigen::ArrayXXf to_destination = destinations(Eigen::all, { d * 4, (d * 4) + 1 }) - population(Eigen::all, { 1,2 });
		Eigen::ArrayXf dist = to_destination.rowwise().norm().array();

		Eigen::ArrayXf head_x = to_destination.col(0) / (dist + epsilon);
		Eigen::ArrayXf head_y = to_destination.col(1) / (dist + epsilon);

		// reinsert headings into population of those not at destination yet
		// set speed to 0.5
		ArrayXXb cond(population.rows(), 2);
		cond << (population.col(11) == d), (population.col(12) == 0);

		population(select_rows(cond), { 3 }) = head_x(select_rows(cond)) * travel_speed;
		population(select_rows(cond), { 4 }) = head_y(select_rows(cond)) * travel_speed;

	}

		

}

/*-----------------------------------------------------------*/
/*                Check who is at destination                */
/*-----------------------------------------------------------*/
void COVID_SIM::check_at_destination(Eigen::ArrayXXf &population, Eigen::ArrayXXf destinations, double wander_factor)
{
	/*check who is at their destination already

	Takes subset of population with active destination and
	tests who is at the required coordinates.Updates at destination
	column for people at destination.

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	destinations : ndarray
	the array containing all destinations information

	wander_factor : int or float
	defines how far outside of 'wander range' the destination reached
	is triggered
	*/

	// how many destinations are active
	Eigen::ArrayXf dests = population.col(11);
	std::vector<float> active_dests(dests.rows()); Eigen::Map<Eigen::ArrayXf>(&active_dests[0], dests.rows(), 1) = dests;
	unique_elements(active_dests);

	// see who is at destination
	for (int d : active_dests) {

		// pick x and y columns for given d
		Eigen::ArrayXf dest_x = destinations(Eigen::all, { d * 4 });
		Eigen::ArrayXf dest_y = destinations(Eigen::all, { (d * 4) + 1 });

		// see who arrived at destination and filter out who already was there
		ArrayXXb cond(population.rows(), 4);
		cond << ((population.col(1) - dest_x).abs() < (wander_factor)),
				((population.col(2) - dest_y).abs() < (wander_factor)),
				(population.col(12) == 0), (population.col(11) == d);

		Eigen::ArrayXXf at_dest = population(select_rows(cond), Eigen::all);

		if (at_dest.rows() > 0) {
			// mark those as arrived
			population(select_rows(cond), { 12 }) = 1;
		}

	}
}

/*-----------------------------------------------------------*/
/*            Keeps arrivals within wander range             */
/*-----------------------------------------------------------*/
void COVID_SIM::keep_at_destination(Eigen::ArrayXXf &population,
	std::vector<double> lb_environments, std::vector<double> ub_environments,
	double wall_buffer, double bounce_buffer)
{
	/*keeps those who have arrived, within wander range

	Function that keeps those who have been marked as arrived at their
	destination within their respective wander ranges

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	destination_bounds : list or tuple
	defines bounds for the location the individual will be roam in when sent
	there.format : [xmin, ymin, xmax, ymax]
	*/ 

	// how many destinations are active
	Eigen::ArrayXf dests = population.col(11);
	std::vector<float> active_dests(dests.rows()); Eigen::Map<Eigen::ArrayXf>(&active_dests[0], dests.rows(), 1) = dests;
	unique_elements(active_dests);

	for (int d : active_dests) {
		// see who is marked as arrived
		ArrayXXb cond(population.rows(), 2);
		cond << (population.col(12) == 1), (population.col(11) == d);
		Eigen::ArrayXXf arrived = population(select_rows(cond), Eigen::all);

		Eigen::ArrayXf ids = arrived.col(0); // find unique IDs of arrived persons

		// check if there are those out of bounds
		double i_xlower = lb_environments[(2 * d)]; double i_xupper = ub_environments[(2 * d)];
		double i_ylower = lb_environments[(2 * d) + 1]; double i_yupper = ub_environments[(2 * d) + 1];

		Eigen::ArrayXXf _xbounds(arrived.rows(), 2), _ybounds(arrived.rows(), 2);
		double buffer = 0.0;

		_xbounds << Eigen::ArrayXf::Ones(arrived.rows(), 1) * (i_xlower + buffer),
					Eigen::ArrayXf::Ones(arrived.rows(), 1) * (i_xupper - buffer);

		_ybounds << Eigen::ArrayXf::Ones(arrived.rows(), 1) * (i_ylower + buffer),
					Eigen::ArrayXf::Ones(arrived.rows(), 1) * (i_yupper - buffer);

		arrived = update_wall_forces(arrived, _xbounds, _ybounds, wall_buffer, bounce_buffer);

		// reinsert into population
		population(select_rows(cond), Eigen::all) = arrived;
	}

}

/*-----------------------------------------------------------*/
/*                 Clear destination markers                 */
/*-----------------------------------------------------------*/
void COVID_SIM::reset_destinations(Eigen::ArrayXXf &population, std::vector<int> Ids)
{
	/*clears destination markers

	Function that clears all active destination markers from the population

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	ids : ndarray or list
	array containing the id's of the population members that need their
	destinations reset
	*/

	if (Ids.size() == 0) {
		// if ids empty, reset everyone
		population.col(11) = 0;
	}
}
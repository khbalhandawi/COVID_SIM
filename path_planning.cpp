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
 \file   path_planning.cpp
 \brief  Methods related to goal-directed traveling behaviour and path planning (implementation)
 \author Khalil Al Handawi
 \date   2010-08-25
 \see    path_planning.h
 */

#include "path_planning.h"

/*-----------------------------------------------------------*/
/*                  Send patient to location                 */
/*-----------------------------------------------------------*/
void go_to_location(Eigen::VectorXi Choices, Eigen::ArrayXXd &patients, Eigen::ArrayXXd &destinations, vector<double> location_bounds, int dest_no)
{
	/*sends patient to defined location

	Function that takes a patient an destination, and sets the location
	as active for that patient.

	Keyword arguments
	---------------- -
	population : ndarray
	the array containing all the population information

	destinations : ndarray
	the array containing all destinations information

	location_bounds : list or tuple
	defines bounds for the location the patient will be roam in when sent
	there.format : [xmin, ymin, xmax, ymax]

	dest_no : int
	the location number, used as index for destinations array if multiple possible
	destinations are defined`.


	TODO : vectorize

	*/

	vector<double> outputs = get_motion_parameters(location_bounds[0], location_bounds[1], location_bounds[2], location_bounds[3]);
	
	double x_center, y_center, x_wander, y_wander;
	x_center = outputs[0];
	y_center = outputs[1];
	x_wander = outputs[2];
	y_wander = outputs[3];

	patients.col(13) = x_wander;
	patients.col(14) = y_wander;

	destinations(Choices, { (dest_no - 1) * 2 }) = x_center;
	destinations(Choices, { ((dest_no - 1) * 2) + 1 }) = y_center;

	patients(Choices, { 11 }) = dest_no; // set destination active
}

/*-----------------------------------------------------------*/
/*                Set population destination                 */
/*-----------------------------------------------------------*/
void set_destination(Eigen::ArrayXXd &population, Eigen::ArrayXXd destinations, double travel_speed)
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

	// how many destinations are active
	Eigen::ArrayXd dests = population(select_rows(population.col(11) != 0), { 11 });
	vector<double> active_dests(dests.rows()); Eigen::Map<Eigen::ArrayXd>(&active_dests[0], dests.rows(), 1) = dests;
	unique_elements(active_dests);

	// set destination
	for (int d : active_dests) {

		// pick x and y columns for given d
		Eigen::ArrayXXd to_destination = destinations(Eigen::all, { (d - 1) * 2, ((d - 1) * 2) + 1 }) - population(Eigen::all, { 1,2 });
		Eigen::ArrayXd dist = to_destination.rowwise().norm().array();

		Eigen::ArrayXd head_x = to_destination.col(0) / dist;
		Eigen::ArrayXd head_y = to_destination.col(1) / dist;

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
void check_at_destination(Eigen::ArrayXXd &population, Eigen::ArrayXXd destinations, double wander_factor)
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
	Eigen::ArrayXd dests = population(select_rows(population.col(11) != 0), { 11 });
	vector<double> active_dests(dests.rows()); Eigen::Map<Eigen::ArrayXd>(&active_dests[0], dests.rows(), 1) = dests;
	unique_elements(active_dests);

	// see who is at destination
	for (int d : active_dests) {

		// pick x and y columns for given d
		Eigen::ArrayXd dest_x = destinations(Eigen::all, { (d - 1) * 2 });
		Eigen::ArrayXd dest_y = destinations(Eigen::all, { ((d - 1) * 2) + 1 });

		// see who arrived at destination and filter out who already was there
		ArrayXXb cond(population.rows(), 3);
		cond << ((population.col(1) - dest_x).abs() < (population.col(13) * wander_factor)),
				((population.col(2) - dest_y).abs() < (population.col(14) * wander_factor)),
				(population.col(12) == 0);

		Eigen::ArrayXXd at_dest = population(select_rows(cond), Eigen::all);

		if (at_dest.rows() > 0) {
			// mark those as arrived
			population(select_rows(cond), { 12 }) = 1;
		}

	}
}

/*-----------------------------------------------------------*/
/*            Keeps arrivals within wander range             */
/*-----------------------------------------------------------*/
void keep_at_destination(Eigen::ArrayXXd &population, vector<double> destination_bounds)
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
	Eigen::ArrayXd dests = population(select_rows(population.col(11) != 0), { 11 });
	vector<double> active_dests(dests.rows()); Eigen::Map<Eigen::ArrayXd>(&active_dests[0], dests.rows(), 1) = dests;
	unique_elements(active_dests);

	for (int d : active_dests) {
		// see who is marked as arrived
		ArrayXXb cond(population.rows(), 2);
		cond << (population.col(12) == 1), (population.col(11) == d);
		Eigen::ArrayXXd arrived = population(select_rows(cond), Eigen::all);

		Eigen::ArrayXd ids = arrived.col(0); // find unique IDs of arrived persons

		// check if there are those out of bounds
		double i_xlower = destination_bounds[0]; double i_xupper = destination_bounds[2];
		double i_ylower = destination_bounds[1]; double i_yupper = destination_bounds[3];

		Eigen::ArrayXXd _xbounds(arrived.rows(), 2), _ybounds(arrived.rows(), 2);
		double buffer = 0.0;

		_xbounds << Eigen::ArrayXd::Ones(arrived.rows(), 1) * (i_xlower + buffer),
					Eigen::ArrayXd::Ones(arrived.rows(), 1) * (i_xupper - buffer);

		_ybounds << Eigen::ArrayXd::Ones(arrived.rows(), 1) * (i_ylower + buffer),
					Eigen::ArrayXd::Ones(arrived.rows(), 1) * (i_yupper - buffer);

		arrived = update_wall_forces(arrived, _xbounds, _ybounds);

		// reinsert into population
		population(select_rows(cond), Eigen::all) = arrived;
	}

}

/*-----------------------------------------------------------*/
/*                 Clear destination markers                 */
/*-----------------------------------------------------------*/
void reset_destinations(Eigen::ArrayXXd &population, vector<int> Ids)
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
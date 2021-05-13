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
 \file   io_utilities.cpp
 \brief  Blackbox support functons (implementation)
 \author Khalil Al Handawi
 \date   2010-08-25
 \see    io_utilities.h
 */


#include "io_utilities.h"

#include <map>
#include <iostream>
#include <sstream>
#include <iterator>

/*-----------------------------------------------------------*/
/*            Scale a variable between 0 and 1               */
/*-----------------------------------------------------------*/
double IO_BB::scaling(double x, double l, double u, int type) {

	double x_out;

	if (type == 1) {
		// scale
		x_out = (x - l) / (u - l);
	} 
	else if (type == 2) {
		// unscale
		x_out = l + x*(u - l);
	}

	return x_out;
}

/*-----------------------------------------------------------*/
/*                  avaerage of std::vector                  */
/*-----------------------------------------------------------*/
double IO_BB::average(double* values, int n_samples) {
	double sum = 0;
	for (size_t i = 0; i < n_samples; i++) {
		sum += values[i];
	}

	double mean = sum / (double)n_samples;

	return mean;
}

/*-----------------------------------------------------------*/
/*                reliability of std::vector                 */
/*-----------------------------------------------------------*/
double IO_BB::reliability(double* values, int n_samples) {
	int count = 0;
	for (size_t i = 0; i < n_samples; i++) {

		if (values[i] <= 0) {
			count += 1;
		}

	}

	double rel = (double)count / (double)n_samples;

	return rel;
}

/*-----------------------------------------------------------*/
/*              Lookup a specific past point                 */
/*-----------------------------------------------------------*/
std::vector<double> IO_BB::lookupHistory(int tag, int n_evals, double **history) {

	// number of points
	int lookup;
	double pt_tag, x1, x2, x3, mean_f, mean_c, p_value;

	for (int i = 0; i < n_evals + 1; ++i)
	{
		lookup = static_cast<int> (history[i][0]);

		if (tag == lookup) {
			pt_tag = history[i][0];
			x1 = history[i][1];
			x2 = history[i][2];
			x3 = history[i][3];
			mean_f = history[i][4];
			mean_c = history[i][5];
			p_value = history[i][6];
			break;
		}
	}

	std::vector<double> output = { pt_tag, x1, x2, x3, mean_f, mean_c, p_value };
	return output;
}
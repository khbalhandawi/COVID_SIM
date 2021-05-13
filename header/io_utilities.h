#pragma once


#include "Configuration.h"
#include "simulation.h"
#include <fstream>

#ifndef IO_UTILITIES_H_H
#define IO_UTILITIES_H_H

namespace IO_BB {

	/*-----------------------------------------------------------*/
	/*            Scale a variable between 0 and 1               */
	/*-----------------------------------------------------------*/
	double scaling(double value, double lb, double ub, int type=1);

	/*-----------------------------------------------------------*/
	/*                  avaerage of std::vector                  */
	/*-----------------------------------------------------------*/
	double average(double* values, int n_samples);

	/*-----------------------------------------------------------*/
	/*                reliability of std::vector                 */
	/*-----------------------------------------------------------*/
	double reliability(double* values, int n_samples);

	/*-----------------------------------------------------------*/
	/*              Lookup a specific past point                 */
	/*-----------------------------------------------------------*/
	std::vector<double> lookupHistory(int tag, int n_eval, double **history);

	/*-----------------------------------------------------------*/
	/*               print a std::vector to file                 */
	/*-----------------------------------------------------------*/
	template <typename T>
	void writeToFile(std::vector<T> values, std::ofstream* output_file) {
		// Convert int to ostring stream
		std::ostringstream oss;
		oss.precision(20);
		oss << std::fixed;
		if (!values.empty())
		{
			// Convert all but the last element to avoid a trailing ","
			std::copy(values.begin(), values.end() - 1,
				std::ostream_iterator<T>(oss, ","));

			// Now add the last element with no delimiter
			oss << values.back();
		}

		output_file->precision(20);
		// Write ostring stream to file

		if (output_file->is_open())
		{
			(*output_file) << std::fixed << oss.str() << '\n';
		}
	}

}

#endif
// IO_UTILITIES_H_H
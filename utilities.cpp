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
 \file   utilities.cpp
 \brief  Utility functions used as part of simulation (implementation)
 \author Khalil Al Handawi
 \date   2010-08-25
 \see    utilities.h
 */

#include "utilities.h"
#include <windows.h>

/*-----------------------------------------------------------*/
/*                    Select block of data                   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf Select_block(Eigen::ArrayXXf input, std::vector<int> cols, int n_rows, Eigen::ArrayXf cond_vector, int cond_value, string cond_type)
{
	Eigen::ArrayXf zero_vector = Eigen::ArrayXf::Zero(n_rows, 1);
	Eigen::ArrayXXf output(n_rows, cols.size());
	Eigen::ArrayXf col;

	for (size_t n = 0; n < cols.size(); ++n) {

		if (cond_type == "==") { col = (cond_vector == cond_value).select(input.col(cols[n]), zero_vector); }
		else if (cond_type == ">=") { col = (cond_vector >= cond_value).select(input.col(cols[n]), zero_vector); }
		else if (cond_type == "<=") { col = (cond_vector <= cond_value).select(input.col(cols[n]), zero_vector); }
		else if (cond_type == ">") { col = (cond_vector > cond_value).select(input.col(cols[n]), zero_vector); }
		else if (cond_type == "<") { col = (cond_vector < cond_value).select(input.col(cols[n]), zero_vector); }

		output.col(n) = col;
	}

	return output;
}

/*-----------------------------------------------------------*/
/*                    Select rows of data                    */
/*-----------------------------------------------------------*/
vector<int> select_rows(ArrayXXb cond)
{	

	int n_rows = cond.rows();
	vector<int> keep_rows;

	for (int i = 0; i < n_rows; ++i) {

		if ((cond.row(i) == 1).all()) {
			keep_rows.push_back(i);
		}
	}

	return keep_rows;
}

/*-----------------------------------------------------------*/
/*                 Select rows of data (any)                 */
/*-----------------------------------------------------------*/
vector<int> select_rows_any(ArrayXXb cond)
{

	int n_rows = cond.rows();
	vector<int> keep_rows;

	for (int i = 0; i < n_rows; ++i) {

		if ((cond.row(i) == 1).any()) {
			keep_rows.push_back(i);
		}
	}

	return keep_rows;
}

/*-----------------------------------------------------------*/
/*                  Select elements of data                  */
/*-----------------------------------------------------------*/
tuple<vector<int>, vector<int>> select_rows_cols(ArrayXXb conds)
{

	int n_rows = conds.rows();
	int n_cols = conds.cols();
	vector<int> keep_rows;
	vector<int> keep_cols;

	for (int i = 0; i < n_rows; ++i) {

		for (int j = 0; j < n_cols; ++j) {

			if (conds(i, j) == 0) {
				keep_rows.push_back(i);
				keep_cols.push_back(j);
			}

		}
	}

	return { keep_rows, keep_cols };
}

/*-----------------------------------------------------------*/
/*              Function for repeating a vector              */
/*-----------------------------------------------------------*/
Eigen::ArrayXf repeat(Eigen::ArrayXf A, int n_times)
{
	Eigen::ArrayXf B(A.rows()*(n_times));

	for (int i = 0; i < A.rows(); i++) {

		double cur = A(i);

		for (int j = 0; j < n_times; j++) {
			B(n_times * i + j) = cur;
		}
	}

	return B;
}

/*-----------------------------------------------------------*/
/*        Function for generating a linspaced vector         */
/*-----------------------------------------------------------*/
// Create a vector of evenly spaced numbers.
vector<double> range(double min, double max, size_t N)
{
	vector<double> range;
	double delta = (max - min) / double(N - 1);

	for (int i = 0; i < N; i++) {
		range.push_back(min + i * delta);
	}

	return range;
}

/*-----------------------------------------------------------*/
/*      Function for generating a sequence of integers       */
/*-----------------------------------------------------------*/
vector<int> sequence(int min, int max, int skip)
{
	vector<int> sequence;
	int e = min;
	int i = 0;

	while (e < max) {
		sequence.push_back(e);
		i++;
		e = min + i * skip;
	}

	return sequence;
}

/*-----------------------------------------------------------*/
/*    Function for creating a directory if non-existent      */
/*-----------------------------------------------------------*/
void check_folder(string folder)
{
	/*check if folder exists, make if not present*/

	string dir = folder + "\\";

	DWORD const ftyp = GetFileAttributesA(folder.c_str());

	if ((ftyp != INVALID_FILE_ATTRIBUTES) && (ftyp & FILE_ATTRIBUTE_DIRECTORY)) {
		// printf("%s is a directory\n", folder); // this is a directory!
	}
	else {
		// printf("%s is not a directory, creating ...\n", folder); // this is not a directory!
		_mkdir(dir.c_str());
	}
}

/*-----------------------------------------------------------*/
/*    Function for computing the pairwise norm of a vector   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf pairwise_dist(Eigen::ArrayXXf a)
{
	Eigen::ArrayXf D2 = a.rowwise().squaredNorm(); // hurts precision
	Eigen::ArrayXXf dist = D2.rowwise().replicate(a.rows()) + D2.transpose().colwise().replicate(a.rows());
	Eigen::ArrayXXf twoAB = 2.*a.matrix()*a.matrix().transpose();
	dist -= twoAB; // needs a square root

	return dist;
}

/*-----------------------------------------------------------*/
/* Function for computing the pairwise difference of a vector*/
/*-----------------------------------------------------------*/
Eigen::ArrayXXf pairwise_diff(Eigen::ArrayXf a)
{
	Eigen::ArrayXXf dist = a.transpose().colwise().replicate(a.rows()) - a.rowwise().replicate(a.rows());
	return dist;
}

/*-----------------------------------------------------------*/
/*                 Fast inverse square-root                  */
/*-----------------------------------------------------------*/
// See: http://en.wikipedia.org/wiki/Fast_inverse_square_root
double invSqrt(double x)
{
	double halfx = 0.5 * x;
	double y = x;
	long i = *(long*)&y;
	i = 0x5f3759d - (i>>1);
	y = *(double*)&i;
	y = y * (1.5 - (halfx * y * y));
	y = y * (1.5 - (halfx * y * y)); // second iteration

	return y;
}

/*-----------------------------------------------------------*/
/*      Function for mapping Eigen Array to std::vector      */
/*-----------------------------------------------------------*/
vector<double> to_std_vector(Eigen::MatrixXd a)
{

	vector<double> out_i;

	for (int j = 0; j < a.rows(); j++) {
		out_i.push_back(a.row(j)[0]);
	}


	return out_i;
}

/*-----------------------------------------------------------*/
/*                     Slice a std::vector                   */
/*-----------------------------------------------------------*/
vector<int> slice_u(vector<int> const &v, int m, int n)
{
	auto first = v.cbegin() + m;
	auto last = v.cbegin() + n;
	
	vector<int> vec(first, last);
	return vec;
}

/*-----------------------------------------------------------*/
/*               Unique elements of std::vector              */
/*-----------------------------------------------------------*/
void unique_elements(vector<float> &v)
{
	// remove consecutive (adjacent) duplicates
	auto last = unique(v.begin(), v.end());
	// v now holds {1 2 1 3 4 5 4 x x x}, where 'x' is indeterminate
	v.erase(last, v.end());
	// sort followed by unique, to remove all duplicates
	sort(v.begin(), v.end()); // {1 1 2 3 4 4 5}
	last = unique(v.begin(), v.end());
	// v now holds {1 2 3 4 5 x x}, where 'x' is indeterminate
	v.erase(last, v.end());
}
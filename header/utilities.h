#pragma once

#include "Defines.h"

#include<vector>
#include<string>
#include<Eigen/Core>

#ifndef UTILITIES_H_H
#define UTILITIES_H_H

namespace COVID_SIM {

	typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb;
	typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> ArrayXXb;

	/*-----------------------------------------------------------*/
	/*                    Select block of data                   */
	/*-----------------------------------------------------------*/
	Eigen::ArrayXXf Select_block(Eigen::ArrayXXf input, std::vector<int> cols, int n_rows, Eigen::ArrayXf cond_vector, int cond_value, std::string cond_type);

API_BEGIN
	/*-----------------------------------------------------------*/
	/*                    Select rows of data                    */
	/*-----------------------------------------------------------*/
	std::vector<int> DLL_API select_rows(ArrayXXb cond);

	/*-----------------------------------------------------------*/
	/*                 Select rows of data (any)                 */
	/*-----------------------------------------------------------*/
	std::vector<int> DLL_API select_rows_any(ArrayXXb cond);
API_END

	/*-----------------------------------------------------------*/
	/*                  Select elements of data                  */
	/*-----------------------------------------------------------*/
	std::tuple<std::vector<int>, std::vector<int>> select_rows_cols(ArrayXXb conds);

	/*-----------------------------------------------------------*/
	/*              Function for repeating a vector              */
	/*-----------------------------------------------------------*/
	Eigen::ArrayXf repeat(Eigen::ArrayXf A, int n_times);

	/*-----------------------------------------------------------*/
	/*        Function for generating a linspaced vector         */
	/*-----------------------------------------------------------*/
	std::vector<double> range(double min, double max, size_t N);

	/*-----------------------------------------------------------*/
	/*      Function for generating a sequence of integers       */
	/*-----------------------------------------------------------*/
	std::vector<int> sequence(int min, int max, int skip = 1);

API_BEGIN
	/*-----------------------------------------------------------*/
	/*    Function for creating a directory if non-existent      */
	/*-----------------------------------------------------------*/
    void DLL_API check_folder(std::string folder = "render");
API_END

	/*-----------------------------------------------------------*/
	/*    Function for computing the pairwise norm of a vector   */
	/*-----------------------------------------------------------*/
	Eigen::ArrayXXf pairwise_dist(Eigen::ArrayXXf a);

	/*-----------------------------------------------------------*/
	/* Function for computing the pairwise difference of a vector*/
	/*-----------------------------------------------------------*/
	Eigen::ArrayXXf pairwise_diff(Eigen::ArrayXf a);

	/*-----------------------------------------------------------*/
	/*                 Fast inverse square-root                  */
	/*-----------------------------------------------------------*/
	// See: http://en.wikipedia.org/wiki/Fast_inverse_square_root
	double invSqrt(double x);

	/*-----------------------------------------------------------*/
	/*      Function for mapping Eigen Array to std::vector      */
	/*-----------------------------------------------------------*/
	std::vector<double> to_std_vector(Eigen::MatrixXd a);

	/*-----------------------------------------------------------*/
	/*                     Slice a std::vector                   */
	/*-----------------------------------------------------------*/
	std::vector<int> slice_u(std::vector<int> const &v, int m, int n);

	/*-----------------------------------------------------------*/
	/*               Unique elements of std::vector              */
	/*-----------------------------------------------------------*/
	void unique_elements(std::vector<float> &v);

}

#endif

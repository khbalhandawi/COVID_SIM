#pragma once

#include <iostream>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <tuple>
#include <direct.h>

using namespace std;

typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb;
typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> ArrayXXb;

/*-----------------------------------------------------------*/
/*                    Select block of data                   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXd Select_block(Eigen::ArrayXXd input, std::vector<int> cols, int n_rows, Eigen::ArrayXd cond_vector, int cond_value, string cond_type);

/*-----------------------------------------------------------*/
/*                    Select rows of data                    */
/*-----------------------------------------------------------*/
vector<int> select_rows(ArrayXXb cond);

/*-----------------------------------------------------------*/
/*                 Select rows of data (any)                 */
/*-----------------------------------------------------------*/
vector<int> select_rows_any(ArrayXXb cond);

/*-----------------------------------------------------------*/
/*                  Select elements of data                  */
/*-----------------------------------------------------------*/
tuple<vector<int>, vector<int>> select_rows_cols(ArrayXXb conds);

/*-----------------------------------------------------------*/
/*              Function for repeating a vector              */
/*-----------------------------------------------------------*/
Eigen::ArrayXd repeat(Eigen::ArrayXd A, int n_times);

/*-----------------------------------------------------------*/
/*        Function for generating a linspaced vector         */
/*-----------------------------------------------------------*/
vector<double> range(double min, double max, size_t N);

/*-----------------------------------------------------------*/
/*      Function for generating a sequence of integers       */
/*-----------------------------------------------------------*/
vector<int> sequence(int min, int max, int skip = 1);

/*-----------------------------------------------------------*/
/*    Function for creating a directory if non-existent      */
/*-----------------------------------------------------------*/
void check_folder(string folder = "render");

/*-----------------------------------------------------------*/
/*    Function for computing the pairwise norm of a vector   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXd pairwise_dist(Eigen::ArrayXXd a);

/*-----------------------------------------------------------*/
/* Function for computing the pairwise difference of a vector*/
/*-----------------------------------------------------------*/
Eigen::ArrayXXd pairwise_diff(Eigen::ArrayXd a);

/*-----------------------------------------------------------*/
/*                 Fast inverse square-root                  */
/*-----------------------------------------------------------*/
// See: http://en.wikipedia.org/wiki/Fast_inverse_square_root
double invSqrt(double x);

/*-----------------------------------------------------------*/
/*      Function for mapping Eigen Array to std::vector      */
/*-----------------------------------------------------------*/
vector<double> to_std_vector(Eigen::MatrixXd a);

/*-----------------------------------------------------------*/
/*                     Slice a std::vector                   */
/*-----------------------------------------------------------*/
vector<int> slice(vector<int> const &v, int m, int n);

/*-----------------------------------------------------------*/
/*               Unique elements of std::vector              */
/*-----------------------------------------------------------*/
void unique_elements(vector<double> &v);
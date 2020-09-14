#pragma once

#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <map>
#include <math.h>
#include <tuple>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "utilities.h"
#include <fstream>

using namespace std;
using namespace std::chrono;

/*-----------------------------------------------------------*/
/*                 Random device initializer                 */
/*-----------------------------------------------------------*/
class RandomDevice {
private:
public:
	unsigned long rand_seed;
	default_random_engine engine;
	uniform_real_distribution<double> distribution_ur;
	uniform_int_distribution<int> distribution_ui;
	normal_distribution<double> distribution_norm;
	/*-----------------------------------------------------------*/
	/*                 Random device Constructor                 */
	/*-----------------------------------------------------------*/
	RandomDevice(unsigned long n);

	/*-----------------------------------------------------------*/
	/*              Uniform distribtuion generator               */
	/*-----------------------------------------------------------*/
	double randUniform(double min, double max);

	/*-----------------------------------------------------------*/
	/*               Uniform integer distribtuion                */
	/*-----------------------------------------------------------*/
	int randUniformInt(int low, int high);

	/*-----------------------------------------------------------*/
	/*               Normal distribtuion generator               */
	/*-----------------------------------------------------------*/
	double randNormal(double mean, double std);

	/*-----------------------------------------------------------*/
	/*      Random uniform distribution for a population         */
	/*-----------------------------------------------------------*/
	Eigen::ArrayXXd uniform_dist(double low, double high, int size_x, int size_y);

	/*-----------------------------------------------------------*/
	/*      Normal uniform distribution for a population         */
	/*-----------------------------------------------------------*/
	Eigen::ArrayXXd normal_dist(double low, double high, int size_x, int size_y);

	/*-----------------------------------------------------------*/
	/*      Randomly select population members (probability)     */
	/*-----------------------------------------------------------*/
	Eigen::ArrayXd Random_choice_prob(int pop_size, double percentage_pop);

	/*-----------------------------------------------------------*/
	/*       Randomly select population members (shuffle)        */
	/*-----------------------------------------------------------*/
	Eigen::VectorXi Random_choice(Eigen::ArrayXd input, int n_choices);

	/*-----------------------------------------------------------*/
	/*                Randomly generate a number                 */
	/*-----------------------------------------------------------*/
	double rand();

	/*-----------------------------------------------------------*/
	/*               Save random generator state                 */
	/*-----------------------------------------------------------*/
	void save_state();

	/*-----------------------------------------------------------*/
	/*               Load random generator state                 */
	/*-----------------------------------------------------------*/
	void load_state();

	/*-----------------------------------------------------------*/
	/*                  Random device Destructor                 */
	/*-----------------------------------------------------------*/
	~RandomDevice();
};


#pragma once

#include "Defines.h"

#include <random>
#include<Eigen/Core>

#ifndef RANDOMDEVICE_H_H
#define RANDOMDEVICE_H_H

namespace COVID_SIM {

API_BEGIN
	/*-----------------------------------------------------------*/
	/*                 Random device initializer                 */
	/*-----------------------------------------------------------*/
	class DLL_API RandomDevice {
	private:
	public:
		unsigned long rand_seed;
		std::default_random_engine engine;
		std::uniform_real_distribution<float> distribution_ur;
		std::uniform_int_distribution<int> distribution_ui;
		std::normal_distribution<float> distribution_norm;
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
		Eigen::ArrayXXf uniform_dist(double low, double high, int size_x, int size_y);

		/*-----------------------------------------------------------*/
		/*      Normal uniform distribution for a population         */
		/*-----------------------------------------------------------*/
		Eigen::ArrayXXf normal_dist(double low, double high, int size_x, int size_y);

		/*-----------------------------------------------------------*/
		/*      Randomly select population members (probability)     */
		/*-----------------------------------------------------------*/
		Eigen::ArrayXf Random_choice_prob(int pop_size, double percentage_pop);

		/*-----------------------------------------------------------*/
		/*       Randomly select population members (shuffle)        */
		/*-----------------------------------------------------------*/
		Eigen::VectorXi Random_choice(Eigen::ArrayXf input, int n_choices);

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
API_END

}

#endif // RANDOMDEVICE_H_H

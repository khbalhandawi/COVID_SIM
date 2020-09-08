#pragma once

#include "utilities.h"
#include "RandomDevice.h"

using namespace std;

/*-----------------------------------------------------------*/
/*                      Update positions                     */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf update_positions(Eigen::ArrayXXf population, double dt = 0.01);

/*-----------------------------------------------------------*/
/*                      Update velocities                    */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf update_velocities(Eigen::ArrayXXf population, double max_speed = 0.3, double dt = 0.01);

/*-----------------------------------------------------------*/
/*                      Update wall forces                   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf update_wall_forces(Eigen::ArrayXXf population, Eigen::ArrayXXf xbounds, Eigen::ArrayXXf ybounds, double wall_buffer = 0.01, double bounce_buffer = 0.005);

/*-----------------------------------------------------------*/
/*                   Update repulsive forces                 */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf update_repulsive_forces(Eigen::ArrayXXf population, double social_distance_factor);

/*-----------------------------------------------------------*/
/*                    Update gravity forces                  */
/*-----------------------------------------------------------*/
tuple<Eigen::ArrayXXf, double> update_gravity_forces(Eigen::ArrayXXf population, double time, double last_step_change, RandomDevice *my_rand, double wander_step_size = 0.01,
	double gravity_strength = 0.1, double wander_step_duration = 0.01);

/*-----------------------------------------------------------*/
/*                  Get motion parameters                    */
/*-----------------------------------------------------------*/
vector<double> get_motion_parameters(double xmin, double ymin, double xmax, double ymax);
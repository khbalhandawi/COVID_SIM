#pragma once

#include "utilities.h"
#include "RandomDevice.h"

using namespace std;

/*-----------------------------------------------------------*/
/*                      Update positions                     */
/*-----------------------------------------------------------*/
Eigen::ArrayXXd update_positions(Eigen::ArrayXXd population, double dt = 0.01);

/*-----------------------------------------------------------*/
/*                      Update velocities                    */
/*-----------------------------------------------------------*/
Eigen::ArrayXXd update_velocities(Eigen::ArrayXXd population, double max_speed = 0.3, double dt = 0.01);

/*-----------------------------------------------------------*/
/*                      Update wall forces                   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXd update_wall_forces(Eigen::ArrayXXd population, Eigen::ArrayXXd xbounds, Eigen::ArrayXXd ybounds, double wall_buffer = 0.01, double bounce_buffer = 0.005);

/*-----------------------------------------------------------*/
/*                   Update repulsive forces                 */
/*-----------------------------------------------------------*/
Eigen::ArrayXXd update_repulsive_forces(Eigen::ArrayXXd population, double social_distance_factor);

/*-----------------------------------------------------------*/
/*                    Update gravity forces                  */
/*-----------------------------------------------------------*/
tuple<Eigen::ArrayXXd, double> update_gravity_forces(Eigen::ArrayXXd population, double time, double last_step_change, RandomDevice *my_rand, double wander_step_size = 0.01,
	double gravity_strength = 0.1, double wander_step_duration = 0.01);

/*-----------------------------------------------------------*/
/*                  Get motion parameters                    */
/*-----------------------------------------------------------*/
vector<double> get_motion_parameters(double xmin, double ymin, double xmax, double ymax);
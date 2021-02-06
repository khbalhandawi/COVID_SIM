#pragma once

#include "utilities.h"
#include "RandomDevice.h"
#ifdef GPU_ACC
#include "CUDA_functions.h"
#endif // GPU_ACC

using namespace std;

/*-----------------------------------------------------------*/
/*                      Update positions                     */
/*-----------------------------------------------------------*/
void update_positions(Eigen::ArrayXXf &population, double dt = 0.01);

#ifdef GPU_ACC
/*-----------------------------------------------------------*/
/*                   Update repulsive forces                 */
/*-----------------------------------------------------------*/
void update_repulsive_forces_cuda(Eigen::ArrayXXf &population_all, double social_distance_factor, CUDA_GPU::Kernels *ABM_cuda);

#else
/*-----------------------------------------------------------*/
/*                   Update repulsive forces                 */
/*-----------------------------------------------------------*/
void update_repulsive_forces(Eigen::ArrayXXf &population_all, double social_distance_factor, Eigen::ArrayXXf &dist_all, bool compute_dist_all = false);
#endif // GPU_ACC

/*-----------------------------------------------------------*/
/*                      Update velocities                    */
/*-----------------------------------------------------------*/
void update_velocities(Eigen::ArrayXXf &population_all, double max_speed = 0.3, double dt = 0.01);

/*-----------------------------------------------------------*/
/*                      Update wall forces                   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf update_wall_forces(Eigen::ArrayXXf population, Eigen::ArrayXXf xbounds, Eigen::ArrayXXf ybounds, double wall_buffer = 0.01, double bounce_buffer = 0.005);

/*-----------------------------------------------------------*/
/*                    Update gravity forces                  */
/*-----------------------------------------------------------*/
void update_gravity_forces(Eigen::ArrayXXf &population, double time, double &last_step_change, RandomDevice *my_rand, double wander_step_size = 0.01,
	double gravity_strength = 0.1, double wander_step_duration = 0.01);

/*-----------------------------------------------------------*/
/*                  Get motion parameters                    */
/*-----------------------------------------------------------*/
vector<double> get_motion_parameters(double xmin, double ymin, double xmax, double ymax);
#pragma once


#include "Configuration.h"
#include "simulation.h"
#include <fstream>

#ifndef IO_BLACKBOX_FUNCTIONS_H_H
#define IO_BLACKBOX_FUNCTIONS_H_H

namespace COVID_SIM {

	/*-----------------------------------------------------------*/
	/*              Post process simulation results              */
	/*-----------------------------------------------------------*/
	std::vector<double> processInput(int i, simulation *sim, std::ofstream *file = nullptr);

	/*-----------------------------------------------------------*/
	/*                    Load configuration                     */
	/*-----------------------------------------------------------*/
	void load_config(Configuration *config, const char *config_file);

}

#endif
// IO_BLACKBOX_FUNCTIONS_H_H
#pragma once

#include "mainwindow.h"

#include "Configuration.h"
#include "Population_trackers.h"

#include <QApplication>

#ifndef VISUALIZER_H_H
#define VISUALIZER_H_H

class visualizer
{
public:
	 /*-----------------------------------------------------------*/
	 /*                    Start a Qt thread                      */
	 /*-----------------------------------------------------------*/
	 std::unique_ptr<MainWindow> visualizer::start_qt(COVID_SIM::Configuration Config);

	/*-----------------------------------------------------------*/
	/*                     Update Qt window                      */
	/*-----------------------------------------------------------*/
	void visualizer::update_qt(Eigen::ArrayXXf population, 
		int frame, double computation_time, std::unique_ptr<MainWindow> &mainWindow, COVID_SIM::Population_trackers *pop_tracker, COVID_SIM::Configuration *Config);

	/*-----------------------------------------------------------*/
	/*                        Constructor                        */
	/*-----------------------------------------------------------*/
	visualizer();

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~visualizer();
};

#endif // VISUALIZER_H_H
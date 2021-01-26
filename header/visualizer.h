#pragma once

#include "utilities.h"
#include "RandomDevice.h"
#include "Configuration.h"
#include "Population_trackers.h"
#include <algorithm>
#include <thread>
#ifndef _N_QT
#include <QApplication>
#include "mainwindow.h"
#endif _N_QT
using namespace std;

class visualizer
{
public:
#ifndef _N_QT
	 /*-----------------------------------------------------------*/
	 /*                    Start a Qt thread                      */
	 /*-----------------------------------------------------------*/
	 std::unique_ptr<MainWindow> visualizer::start_qt(Configuration Config);

	/*-----------------------------------------------------------*/
	/*                     Update Qt window                      */
	/*-----------------------------------------------------------*/
	void visualizer::update_qt(Eigen::ArrayXXf population, 
		int frame, float R0, double computation_time, std::unique_ptr<MainWindow> &mainWindow);
#endif _N_QT
	/*-----------------------------------------------------------*/
	/*                        Constructor                        */
	/*-----------------------------------------------------------*/
	visualizer();

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~visualizer();
};

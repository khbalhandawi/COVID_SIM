#pragma once

#include "utilities.h"
#include "RandomDevice.h"
#include "Configuration.h"
#include "Population_trackers.h"
#include <algorithm>
#include <thread>
#include <QApplication>
#include "mainwindow.h"

using namespace std;

class visualizer
{
public:

	 /*-----------------------------------------------------------*/
	 /*                    Start a Qt thread                      */
	 /*-----------------------------------------------------------*/
	 std::unique_ptr<MainWindow> visualizer::start_qt(Configuration Config);

	/*-----------------------------------------------------------*/
	/*                     Update Qt window                      */
	/*-----------------------------------------------------------*/
	void visualizer::update_qt(Eigen::ArrayXXf population, 
		int frame, std::unique_ptr<MainWindow> &mainWindow);

	/*-----------------------------------------------------------*/
	/*                        Constructor                        */
	/*-----------------------------------------------------------*/
	visualizer();

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~visualizer();
};

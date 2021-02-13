#pragma once

#include "simulation.h"
#include "utilities.h"

#include <QObject>
#include <vector>

class Worker : public QObject
{
	Q_OBJECT

public:
	Worker(COVID_SIM::simulation *sim_in);
	~Worker();

	COVID_SIM::simulation *sim;
	COVID_SIM::ArrayXXb cond;
	int i; // frame counter

	/*-----------------------------------------------------------*/
	/*             Extract time step results for UI              */
	/*-----------------------------------------------------------*/
	void export_results();

	/*-----------------------------------------------------------*/
	/*              Post process simulation results              */
	/*-----------------------------------------------------------*/
	void processOutputs();

public slots:
	void doWork();
	void setICValue(int IC);
	void setSDValue(int SD);
	void setTCValue(int TC);

signals:
	void resultReady(QVector<double> x0, QVector<double> y0,
		QVector<double> x1, QVector<double> y1,
		QVector<double> x2, QVector<double> y2,
		QVector<double> x3, QVector<double> y3,
		QVector<double> x4, QVector<double> y4,
		int frame, float R0, float computation_time,
		QVector<double> x_lower, QVector<double> y_lower,
		QVector<double> x_upper, QVector<double> y_upper);
	void simulation_finished();
	void time_step_finished();
	void worker_initialized();
};

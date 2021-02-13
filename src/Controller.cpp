#include "Controller.h"
#include "Worker.h"

#include <QThread>

Controller::Controller(QObject *parent)
	: QObject(parent)
{
	Worker *worker = new Worker(this);
	worker->moveToThread(&workerThread);
	connect(&workerThread, &QThread::finished, worker, &QObject::deleteLater);
	connect(this, &Controller::operate, worker, &Worker::doWork);
	connect(worker, &Worker::resultReady, this, &Controller::handleResults);
	workerThread.start();
}

Controller::~Controller() 
{
	workerThread.quit();
	workerThread.wait();
}
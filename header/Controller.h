#pragma once

#include <QObject>
#include <QThread>

class Controller : public QObject
{
	Q_OBJECT 
		QThread workerThread;
public:
	Controller(QObject *parent);
	~Controller();
public slots:
	void handleResults(const QString &);
signals:
	void operate(const QString &);
};
#
#  QCustomPlot Plot Examples
#

QT       += core gui sql
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = COVID_SIM_UI
TEMPLATE = app

HEADERS += header/visualizer.h \
    header/qcustomplot.h \
    header/mainwindow.h \
	
SOURCES += src/visualizer.cpp \
    src/qcustomplot.cpp \
    src/mainwindow.cpp \
    src/covidsim_ui.cpp \
	
FORMS += mainwindow.ui

win32: INCLUDEPATH += header \
    Include \
DEPENDPATH += header \

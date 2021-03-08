#
#  QCustomPlot Plot Examples
#

QT       += core gui sql
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = COVID_SIM_UI
TEMPLATE = app

HEADERS += header/qcustomplot.h \
    header/mainwindow.h \
    header/worker.h
	
SOURCES += src/qcustomplot.cpp \
    src/mainwindow.cpp \
    src/worker.cpp \
    src/covidsim_ui.cpp \
	
FORMS += mainwindow.ui

win32: INCLUDEPATH += $$PWD/libs\
    header \
    Include
DEPENDPATH += $$PWD/libs \
    header

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/libs/ -lCOVID_sim_lib \
    -L$$PWD/libs/ -lCUDA_functions
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/libs/ -lCOVID_sim_lib \
    -L$$PWD/libs/ -lCUDA_functions_d
else:unix: LIBS += -L$$PWD/libs/ -lCOVID_sim_lib \
    -L$$PWD/libs/ -lCUDA_functions

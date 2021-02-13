#ifndef CONVERT_H_
#define CONVERT_H_

#include "TypeDefs.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
namespace IOFile
{

// Eigen IO
Eigen::MatrixXd readFromfile(const string file);
bool writeTofile(Eigen::MatrixXd matrix, const string file);

} //namesapce IMU

#endif // CONVERT_H_
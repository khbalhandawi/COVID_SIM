#include "Convert.h"
#include "TypeDefs.h"

#include <fstream>
#define _USE_MATH_DEFINES
#include "math.h"

namespace IOFile
{

Eigen::MatrixXd readFromfile(const string file_name)
{
    Eigen::MatrixXd matrix;
    std::vector<double> entries;
    ifstream data(file_name, ios::binary);
    string lineOfData;
    data.precision(20);

    if (data.is_open())
    {
        int i = 0;
        int cols = 0;
        while (data.good())
        {

            getline(data, lineOfData);
            stringstream stream_check(lineOfData);
			stringstream stream(lineOfData);

			int j = 0;
			// Find location of EOF
            while (!stream_check.eof())
            {
                double check;
				stream_check >> check;
                j++;
            }

			// exlude last entry due to EOF
			for (int k = 0; k < (j - 1); ++k)
			{
				double a;
				stream >> a;
				entries.push_back(a);
			}
            cols = j;
            i++;
        }

		// reread last entry since theres no EOF at the end of the data file!!
		stringstream stream_check(lineOfData);
		stringstream stream_erase(lineOfData);
		stringstream stream(lineOfData);

		int j = 0;
		// Find location of EOF
		while (!stream_check.eof())
		{
			double check;
			stream_check >> check;
			j++;
		}

		// pop back the last line added
		for (int k = 0; k < (j - 1); ++k)
		{
			double a;
			stream_erase >> a;
			entries.pop_back();
		}

		// read the entire last line since it does not have a CRLF
		for (int k = 0; k < (j); ++k)
		{
			double a;
			stream >> a;
			entries.push_back(a);
		}

        matrix = Eigen::MatrixXd::Map(&entries[0], cols, i).transpose();

        return matrix;
    }
    else
    {
        cout << "Unable to open file" << std::endl;

        return Eigen::Vector3d::Zero();
    }

}

bool writeTofile(Eigen::MatrixXd matrix, const string file_name)
{
    std::ofstream file(file_name, ios::binary);
    file.precision(20);
    if (file.is_open())
    {
        file << matrix << '\n';
    }
    else
        return false;
    file.close();
    return true;
}

} //IMU
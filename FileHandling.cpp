#include "FileHandling.h"

// Store data in CSV file [https://gist.github.com/zhou-chao/7a7de79de47c652196f1]
void FileHandling::writeCSV(const cv::String & filename, const cv::Mat & m)
{
	std::ofstream myfile;
	myfile.open(filename.c_str());
	myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
	myfile.close();
}

// Give proper name to CSV file. This is usefull for properly naming of numerous training data (suitable if using <=1000 stereo-pairs)
cv::String FileHandling::nameCSV(const cv::String & filenameNoNumber, const int& number) {
	cv::String CSVfile;
	if (number < 10) {
		CSVfile = filenameNoNumber + "00" + std::to_string(number);
	}
	else if (number >= 10 && number < 100) {
		CSVfile = filenameNoNumber + "0" + std::to_string(number);
	}
	else if (number >= 100 && number < 1000) {
		CSVfile = filenameNoNumber + std::to_string(number);
	}
	return CSVfile;
}

cv::String FileHandling::nameCSVwithImageName(const cv::String & filenameNoNumber, const cv::String& imageName) {
	cv::String CSVfile;
		CSVfile = filenameNoNumber + imageName;
	return CSVfile;
}
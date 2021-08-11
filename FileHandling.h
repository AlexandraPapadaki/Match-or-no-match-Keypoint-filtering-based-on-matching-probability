#ifndef FILEHANDLING_H
#define FILEHANDLING_H
#include <iostream>
#include <fstream> 
#include <opencv2/opencv.hpp>

class FileHandling {

public:
	//constructor
	FileHandling(void) {};
	//destructor
	~FileHandling(void) {};

	static void writeCSV(const cv::String & filename, const cv::Mat & m);
	static cv::String nameCSV(const cv::String & filenameNoNumber, const int& number);
	cv::String nameCSVwithImageName(const cv::String & filenameNoNumber, const cv::String& imageName);

};
#endif //FILEHANDLING_H #pragma once


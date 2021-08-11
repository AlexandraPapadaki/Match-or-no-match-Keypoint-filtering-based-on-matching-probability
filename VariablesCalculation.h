#ifndef VARIABLESCALCULATION_H
#define VARIABLESCALCULATION_H
#include <iostream>
#include <fstream> 
#include <opencv2/opencv.hpp>

class VariablesCalculation {

public:
	//constructor
	VariablesCalculation(void) {};
	//destructor
	~VariablesCalculation(void) {};

	static cv::Mat countDominantOrientations(const std::vector<cv::KeyPoint>& keypoints);

	static cv::Mat getGreen(const cv::Mat inputImage, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat & descr);

	static cv::Mat constructVariableMat(const cv::Mat inputImage, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat & descr);
};

#endif //VARIABLESCALCULATION_H #pragma once


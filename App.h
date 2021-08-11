#ifndef APP_H
#define APP_H
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

class App {

public:
	//constructor
	App(void) {};
	//destructor
	~App(void) {};


	static void getTrainedData(const cv::String & TrainingImagesDirectory, const int featAmount);

	static void trainForest(const cv::String & trainedDataPath, const int featAmount);

	static void predictOnImage(const cv::Mat & testImageInput, const cv::String trainedModelXMLFile, const cv::String CSVfileTestImagePredictions, std::vector<cv::KeyPoint> &testKps, cv::Mat &testDesc, cv::Mat &labelsPredicted, const int n_featuresWrtMethod);
};


#endif //APP_H#pragma once

#ifndef MATCHING_H
#define MATCHING_H
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

class Matching {

public:
	//constructor
	Matching(void) {};
	//destructor
	~Matching(void) {};

	static void detectAndDescribe(const cv::Mat & img, std::vector<cv::KeyPoint>& imgKps, cv::Mat & imgDesc, const cv::String & KpsOfImageX);

	static void printMatches(const cv::Mat & source, const cv::Mat & target, const std::vector<cv::KeyPoint>& srcKps, const std::vector<cv::KeyPoint>& tarKps, const std::vector<cv::DMatch>& matches, cv::String const & KindOfMatches);

	static float distancePointEpiline(const cv::Point2f & point, const cv::Vec3f & line);

	static void matchStereoPair(const cv::Mat & source, const cv::Mat & target, const std::vector<cv::KeyPoint>& srcKps, const std::vector<cv::KeyPoint>& tarKps, const cv::Mat & srcDesc, const cv::Mat & tarDesc, std::vector<cv::DMatch>& initialMatches, std::vector<cv::DMatch>& matches, cv::Mat & inliers);

};


#endif //MATCHING_H#pragma once

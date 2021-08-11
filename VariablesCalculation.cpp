#include "VariablesCalculation.h"

// Calculates dominant orientation than SIFT assigned to each keypoint (everything above 80% of the most dominant orientation in histogram)
cv::Mat VariablesCalculation::countDominantOrientations(const std::vector<cv::KeyPoint> &keypoints) {

	cv::Mat domOrientations = cv::Mat::ones(keypoints.size(), 1, CV_64FC1);

	for (int i = 0; (int)i<keypoints.size(); i++)
	{
		if (domOrientations.at<double>(i, 0) == 1) // if point has not been checked before
		{
			cv::Mat dominantsIndex = cv::Mat::zeros(keypoints.size(), 1, CV_64FC1);
			dominantsIndex.at<double>(i, 0) = 1;
			int nDominants = 1;

			for (int j = i + 1; j<keypoints.size(); j++)
			{
				float dist = abs((keypoints[i].pt.x - keypoints[j].pt.x)) + abs((keypoints[i].pt.y - keypoints[j].pt.y));
				if (dist == 0)
				{
					nDominants++;
					dominantsIndex.at<double>(j, 0) = 1;
				}
			}

			for (int k = 0; k<dominantsIndex.rows; k++)
			{
				if (dominantsIndex.at<double>(k, 0) == 1) {
					domOrientations.at<double>(k, 0) = nDominants;
				}
			}
		}
	}
	return domOrientations;
}

// Get green color value for the keypoints
cv::Mat VariablesCalculation::getGreen(const cv::Mat inputImage, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descr) {
	cv::Mat Green = cv::Mat::zeros(descr.rows, 1, CV_64FC1);
	for (int i = 0; i < descr.rows; ++i) {
		Green.at<double>(i, 0) = inputImage.at<cv::Vec3b>(keypoints[i].pt.y, keypoints[i].pt.x)[1];
	}
	return Green;
}

// Construct the featute matrix with the variables' values of the samples
cv::Mat VariablesCalculation::constructVariableMat(const cv::Mat inputImage, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descr) {

	// Extract Green channel values for keypointspoints
	cv::Mat Green = cv::Mat::zeros(descr.rows, 1, CV_64FC1);
	Green = getGreen(inputImage, keypoints, descr);

	// Calculate amount of dominant orientations assigned to each keypoint according to SIFT (>80% of the most dominat orientation in histogram)
	cv::Mat domOrientations = cv::Mat::ones(keypoints.size(), 1, CV_64FC1);
	domOrientations = countDominantOrientations(keypoints);

	// Create feature (variable) matrix
	cv::Mat1f features(keypoints.size(), 8, CV_32F);
	for (int j = 0; j < keypoints.size(); j++) {
		features.at<float>(j, 0) = keypoints[j].pt.x;
		features.at<float>(j, 1) = keypoints[j].pt.y;
		features.at<float>(j, 2) = keypoints[j].octave;
		features.at<float>(j, 3) = keypoints[j].angle;
		features.at<float>(j, 4) = keypoints[j].size;
		features.at<float>(j, 5) = keypoints[j].response;
		features.at<float>(j, 6) = Green.at<double>(j, 0);
		features.at<float>(j, 7) = domOrientations.at<double>(j, 0);
	}
	return features;
}
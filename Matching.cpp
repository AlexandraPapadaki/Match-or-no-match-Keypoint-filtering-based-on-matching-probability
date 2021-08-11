#include "Matching.h"


// Detect and Describe the SIFT keypoints within an image
void Matching::detectAndDescribe(const cv::Mat & img, std::vector<cv::KeyPoint>& imgKps, cv::Mat &imgDesc, const cv::String& KpsOfImageX)
{
	// Keypoint Detection
	cv::Ptr<cv::Feature2D> pImageSIFT = cv::SIFT::create();
	//::xfeatures2d::SIFT::create();
	pImageSIFT->detect(img, imgKps);

	for (int i = 0; i < imgKps.size(); i++)
	{
		cv::circle(img, cv::Point(imgKps[i].pt.x, imgKps[i].pt.y), 9, cv::Scalar(0, 0, 255), 1);
	}

	cv::Mat img_down;
	const int IMAGE_DOWNSAMPLE = 1;
	cv::resize(img, img_down, img.size() / IMAGE_DOWNSAMPLE);
//	cv::imshow(KpsOfImageX, img_down);
	cv::imwrite(KpsOfImageX + ".png", img);

	// Keypoint description
	pImageSIFT->compute(img, imgKps, imgDesc);
}

// Print the calculated matches
void Matching::printMatches(const cv::Mat & source, const cv::Mat & target, const std::vector<cv::KeyPoint> &srcKps, const std::vector<cv::KeyPoint> &tarKps, const std::vector< cv::DMatch >& matches, cv::String const& KindOfMatches)
{
	cv::Mat img_matches;
	drawMatches(source, srcKps, target, tarKps,
		matches, img_matches, cv::Scalar(0,255,0), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	// downsample the image to speed up processing
	cv::Mat img_matches_down;
	const int IMAGE_DOWNSAMPLE = 1;
	cv::resize(img_matches, img_matches_down, img_matches.size() / IMAGE_DOWNSAMPLE);
//	cv::namedWindow(KindOfMatches, 1);
//	imshow(KindOfMatches, img_matches_down);
	cv::imwrite("FinalMatches.png", img_matches_down);
}

// Calculate distance of a point from the corresponding epipolar line
float Matching::distancePointEpiline(const cv::Point2f& point, const cv::Vec3f& line)
{
	//Line is given as a*x + b*y + c = 0
	return std::abs(line[0] * point.x + line[1] * point.y + line[2])
		/ std::sqrt(line[0] * line[0] + line[1] * line[1]);
}

// Pairwise match a stereo pair and filter matches to obtain reliable ones
void Matching::matchStereoPair(const cv::Mat & source, const cv::Mat & target, const std::vector<cv::KeyPoint> &srcKps, const std::vector<cv::KeyPoint> &tarKps, const cv::Mat &srcDesc, const cv::Mat &tarDesc, std::vector< cv::DMatch >& initialMatches, std::vector< cv::DMatch >& matches, cv::Mat &inliers)
{
	// Apply Brute-force bi-directional (from source to target image (ST) and target to source(TS)) matching, keeping 2 NN
	std::vector<std::vector<cv::DMatch>> matches_initialST, matches_initialTS;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
	matcher->knnMatch(srcDesc, tarDesc, matches_initialST, 2);
	matcher->knnMatch(tarDesc, srcDesc, matches_initialTS, 2);

	// Protect initial ST matches
	for (int i = 0; i < (int)matches_initialST.size(); i++) {
		initialMatches.push_back(matches_initialST[i][0]);
	}

	// Apply filters to obtain reliable matches
	// Apply NN ratio to matches
	double ratio = 0.9;
	std::vector<cv::DMatch> matches_RatioST, matches_RatioTS;
	for (int i = 0; i < matches_initialST.size(); i++)
	{
		if (matches_initialST[i][0].distance < ratio * matches_initialST[i][1].distance)
			matches_RatioST.push_back(matches_initialST[i][0]);
	}

	for (int i = 0; i < matches_initialTS.size(); i++)
	{
		if (matches_initialTS[i][0].distance < ratio * matches_initialTS[i][1].distance)
			matches_RatioTS.push_back(matches_initialTS[i][0]);
	}

	// Apply symmetry test to matches to get cross-checked matches
	for (int i = 0; i<matches_RatioST.size(); i++)
	{
		for (int j = 0; j<matches_RatioTS.size(); j++)
		{
			if (matches_RatioST[i].queryIdx == matches_RatioTS[j].trainIdx && matches_RatioTS[j].queryIdx == matches_RatioST[i].trainIdx)
			{
				matches.push_back(cv::DMatch(matches_RatioST[i].queryIdx, matches_RatioST[i].trainIdx, matches_RatioST[i].distance));
				break;
			}
		}
	}

	// Prepare keypoints to be fed to RANSAC
	std::vector<cv::Point2f> imgSrcPts, imgTarPts;
	for (int id = 0; id < matches.size(); id++) {
		imgSrcPts.push_back(srcKps[matches[id].queryIdx].pt);
		imgTarPts.push_back(tarKps[matches[id].trainIdx].pt);
	}

	// Apply RANSAC to get inlier matches
	cv::Mat fundamentalMatrix;
	fundamentalMatrix = findFundamentalMat(imgSrcPts, imgTarPts, CV_FM_RANSAC, 2, 0.99, inliers);

	// Find RANSAC inlier matches
	std::vector<cv::DMatch> RANSAC_matches;
	for (int id = 0; id < matches.size(); id++) {
		if (inliers.at<uchar>(id) == 1) {
			RANSAC_matches.push_back(matches[id]);
		}
	}

	// Draw RANSAC matches on images
//	cv::String dirPredKps = "ImagesWithPredictedKps";
//	std::vector<cv::String> fn;
//	glob(dirPredKps, fn, true);
//	cv::Mat imgKpsPredSrc = imread(fn[0]);
//	cv::Mat imgKpsPredTar = imread(fn[1]);
//	for (int i = 0; i < imgSrcPts.size(); i++) {
//		cv::circle(imgKpsPredSrc, cv::Point(srcKps[matches[i].queryIdx].pt.x, srcKps[matches[i].queryIdx].pt.y), 9, cv::Scalar(0, 255, 0), -2);
//	}
//	for (int i = 0; i < imgTarPts.size(); i++) {
//		cv::circle(imgKpsPredTar, cv::Point(tarKps[matches[i].queryIdx].pt.x, tarKps[matches[i].queryIdx].pt.y), 9, cv::Scalar(0, 255, 0), -2);
//	}
//	cv::imwrite("FinalSrc.png", imgKpsPredSrc);
//	cv::imwrite("FinalTar.png", imgKpsPredTar);


	// Apply epipolar geometry test
	std::vector<cv::Vec3f> epipLinesImgSrc, epipLinesImgTar;
	cv::computeCorrespondEpilines(imgSrcPts, 1, fundamentalMatrix, epipLinesImgTar);
	cv::computeCorrespondEpilines(imgTarPts, 2, fundamentalMatrix, epipLinesImgSrc);

	int outlEpip = 0;
	int inlEpip = 0;
	for (size_t i = 0; i < imgSrcPts.size(); i++)
	{
		if (distancePointEpiline(imgSrcPts[i], epipLinesImgSrc[i]) >2 ||
			distancePointEpiline(imgTarPts[i], epipLinesImgTar[i]) >2)
		{
			if (inliers.at<uchar>(i) == 1) {
				inliers.at<uchar>(i) = 0;
				outlEpip = outlEpip + 1;
			}

		}
	}

	// Find inlier matches according to the epipolar criterion
	std::vector<cv::DMatch> Epip_matches;
	for (size_t i = 0; i < imgSrcPts.size(); i++)
	{
		if (inliers.at<uchar>(i) == 1) {
			inlEpip = inlEpip + 1;
			Epip_matches.push_back(matches[i]);
		}
	}

	// Print out important information
	std::cout << "Amount of reliable matches = " << RANSAC_matches.size() << std::endl;
	std::cout << "Amount of epipolar inliers: " << inlEpip << std::endl;
}

#include "App.h"
#include "Matching.h"
#include "VariablesCalculation.h"
#include "FileHandling.h"

#include <chrono>
#include <opencv2/ml.hpp>
#include <time.h>

// static references for no instances
static Matching matching;
static VariablesCalculation varCalc;
static FileHandling filedeal;

// Shuffle a matrix. Was used for shuffling variables to evaluate their importance
cv::Mat shuffle(const cv::Mat &matrix)
{
	std::vector <int> seeds;
	for (int cont = 0; cont < matrix.rows; cont++)
		seeds.push_back(cont);

	cv::randShuffle(seeds);

	cv::Mat output;
	for (int cont = 0; cont < matrix.rows; cont++)
		output.push_back(matrix.row(seeds[cont]));

	return output;
}

// Pre-training phase. Input training images, obtain reliable matches and get features (variables) and labels
void App::getTrainedData(const cv::String & TrainingImagesDirectory, const int featAmount) {

	// Read training images from directory
	std::vector<cv::String> fn;
	glob(TrainingImagesDirectory, fn, true);
	int countIter = 0;
	cv::Mat imgSrcDesc, imgTarDesc;
	cv::String nameSrc;

	// Iterate over training stereopairs
	for (int i = 0; i < fn.size(); i++)
	{
		countIter += 1;
		// Print out some information for the iterations during the process
		std::cout << "Iteration " << countIter << std::endl;
		std::cout << "Images: " << fn[i] << " and " << fn[i + 1] << std::endl;
		nameSrc = fn[i];
		std::string nameTar = fn[i + 1];

		// Read images
		cv::Mat inputImageSrc = imread(fn[i]);
		if (!inputImageSrc.data) {
			std::cout << "Input image Src not found" << std::endl;
			std::cout << "Press enter to exit" << std::endl;
			std::cin.get();
			exit(-2);
		}

		cv::Mat inputImageTar = imread(fn[i + 1]);
		if (!inputImageTar.data) {
			std::cout << "Input image Tar not found" << std::endl;
			std::cout << "Press enter to exit" << std::endl;
			std::cin.get();
			exit(-2);
		}

		// Downsample to speed up processing
		cv::Mat inputImageSrc_down, inputImageTar_down;
//		cv::String win1 = cv::String("Input source image"), win2 = cv::String("Input target image");
//		cv::namedWindow(win1), cv::namedWindow(win2);
		const int IMAGE_DOWNSAMPLE = 4;
		// Input image Src
		cv::resize(inputImageSrc, inputImageSrc_down, inputImageSrc.size() / IMAGE_DOWNSAMPLE);
//		cv::imshow(win1, inputImageSrc_down);
		// Input image Tar
		cv::resize(inputImageTar, inputImageTar_down, inputImageTar.size() / IMAGE_DOWNSAMPLE);
//		cv::imshow(win2, inputImageTar_down);

		// Clone input images
		cv::Mat imgSrc = inputImageSrc.clone();
		cv::Mat imgTar = inputImageTar.clone();

		// Detect and describe SIFT keypoints for both images of the stereo pair
		std::vector<cv::KeyPoint> imgSrcKps, imgTarKps;
		cv::Mat imgSrcDesc, imgTarDesc;
		matching.detectAndDescribe(imgSrc, imgSrcKps, imgSrcDesc, cv::String("Keypoints in source image"));
		matching.detectAndDescribe(imgTar, imgTarKps, imgTarDesc, cv::String("Keypoints in target image"));

		// Apply pairwise matching to obtain reliable matches
		cv::Mat inlierMatches;
		std::vector<cv::DMatch> matches, initialMatches;
		matching.matchStereoPair(imgSrc, imgSrc, imgSrcKps, imgTarKps, imgSrcDesc, imgTarDesc, initialMatches, matches, inlierMatches);

		// Define Labels for each image
		int n_imgSrc = imgSrcDesc.rows;
		int n_imgTar = imgTarDesc.rows;
		cv::Mat labelsImgSrc = cv::Mat::zeros(n_imgSrc, 1, CV_32S);
		cv::Mat labelsImgTar = cv::Mat::zeros(n_imgTar, 1, CV_32S);
		int n_labImgSrc = 0, n_labImgTar = 0;
		for (int idx = 0; idx < inlierMatches.rows; idx++) {
			if (inlierMatches.at<uchar>(idx) == 1) {
				if (labelsImgSrc.at<int>(matches[idx].queryIdx) == 0) {
					labelsImgSrc.at<int>(matches[idx].queryIdx) = 1;
					n_labImgSrc++;
				}
				if (labelsImgTar.at<int>(matches[idx].trainIdx) == 0) {
					labelsImgTar.at<int>(matches[idx].trainIdx) = 1;
					n_labImgTar++;
				}
			}
		}

		// Convert labels to floats so that they can be merged with features
		cv::Mat dataForCSVimg, labelsImg_Float, dataForCSVimgTar, labelsImgTar_Float, dataForCSV;
		labelsImgSrc.convertTo(labelsImg_Float, CV_32F);
		labelsImgTar.convertTo(labelsImgTar_Float, CV_32F);

		// Define features (variables) for each image
		cv::Mat1f features;
		cv::Mat1f featuresSrc(imgSrcKps.size(), featAmount, CV_32F);
		cv::Mat1f featuresTar(imgSrcKps.size(), featAmount, CV_32F);
		if (featAmount == 8) {
			featuresSrc = varCalc.constructVariableMat(inputImageSrc, imgSrcKps, imgSrcDesc);
			featuresTar = varCalc.constructVariableMat(inputImageTar, imgTarKps, imgTarDesc);
		}
		else if (featAmount == 128) {
			featuresSrc = imgSrcDesc;
			featuresTar = imgTarDesc;
		}
		else { std::cout << "Wrong amount of features in pre-training" << std::endl; }

		// Unbalanced ground truth storage
		cv::Mat featAndLabelsForStorageSRC, featAndLabelsForStorageTAR;
		hconcat(featuresSrc, labelsImg_Float, featAndLabelsForStorageSRC);
		hconcat(featuresTar, labelsImgTar_Float, featAndLabelsForStorageTAR);
		cv::String CSVfiledataUnsortedSRC = filedeal.nameCSV("Ground Truth/Unbalanced S ", countIter);
		filedeal.writeCSV(CSVfiledataUnsortedSRC, featAndLabelsForStorageSRC);
		cv::String CSVfiledataUnsortedTAR = filedeal.nameCSV("Ground Truth/Unbalanced T ", countIter);
		filedeal.writeCSV(CSVfiledataUnsortedTAR, featAndLabelsForStorageTAR);

		vconcat(featuresSrc, featuresTar, features);
		// Merge features and labels of the current pair in one matrix and store them in CSV file
		cv::Mat allLab, featAndLabForCSV;
		vconcat(labelsImg_Float, labelsImgTar_Float, allLab);
		hconcat(features, allLab, featAndLabForCSV);

		// Balance data regarding the two classes (0 = non-matchable, 1 = matchable)
		cv::Mat Data0, balancedData1;
		for (int y = 0; y < featAndLabForCSV.rows; y++) {
			if (featAndLabForCSV.at<float>(y, featAmount) == 1) {
				balancedData1.push_back(featAndLabForCSV.row(y));
			}
			else if (featAndLabForCSV.at<float>(y, featAmount) == 0) {
				Data0.push_back(featAndLabForCSV.row(y));
			}
		}

		// Shuffle data
		cv::Mat shuffledData0 = shuffle(Data0);
		cv::Mat balancedData0;
		for (int y = 0; y < balancedData1.rows; y++) {
			balancedData0.push_back(shuffledData0.row(y));
		}

		// Store tha balanced (Training data) data in a CSV file.
		cv::Mat balanced;
		vconcat(balancedData0, balancedData1, balanced);
		cv::String CSVfiledataBal = filedeal.nameCSV("Training Data/Balanced Pair ", countIter);
		filedeal.writeCSV(CSVfiledataBal, balanced);

		// Print some information regarding the dimentions of the obtained data
		std::cout << "Ground truth data are of dimentions " << featAndLabForCSV.rows << " x " << featAndLabForCSV.cols << std::endl;
		std::cout << "Training data (balanced) are of dimentions " << balanced.rows << " x " << balanced.cols << std::endl;

		// increase i for correct selection of image pairs
		i = i + 1;
	}
}

// Training phase. Construct and train the classifier according to the obtained training data
void App::trainForest(const cv::String & trainedDataPath, const int featAmount){

	// Prepair training data
	// Load files with features and labels of every pair from directory
	std::cout << "Loading training data" << std::endl;
	cv::Mat samplesPerPair, samples, responsesPerPair, responses;
	std::vector<cv::String> fnf;
	glob(trainedDataPath, fnf, true);
	for (int pair = 0; pair < fnf.size(); pair++)
	{
		cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV(fnf[pair], 0);
		// Split features and labels into two matrices namely samples and responses respectively
		samplesPerPair = raw_data->getTrainSamples();
		responsesPerPair = raw_data->getTrainResponses();
		samples.push_back(samplesPerPair);
		responses.push_back(responsesPerPair);
	}
	std::cout << "Loaded training data = " << samples.size() << std::endl;

	// Convert properly the training data for Random Forest classifier
	samples.convertTo(samples, CV_32F);
	responses.convertTo(responses, CV_32S);
	cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE, responses, cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray());

	// Define hyperparameters for Random Forest classifier
	float _priors[] = { 0.4, 0.6 };
	cv::Mat priors(1, 2, CV_32F, _priors);
	cv::Ptr<cv::ml::RTrees> rtree = cv::ml::RTrees::create();
	rtree->setMinSampleCount(2);
	rtree->setRegressionAccuracy(0);
	rtree->setUseSurrogates(false);
	rtree->setPriors(priors);
	rtree->setCalculateVarImportance(true);
	rtree->setActiveVarCount(2);

	// TODO - manually select which architecture to use according to the used trained model
	if (featAmount == 8) {
		rtree->setMaxDepth(5);
		rtree->setTermCriteria({ cv::TermCriteria::MAX_ITER, 5, 0 });
	}
	else if (featAmount == 128) {
		rtree->setMaxDepth(25);
		rtree->setTermCriteria({ cv::TermCriteria::MAX_ITER, 25, 0 });
	}
	else { std::cout << "Wrong amount of features in pre-training" << std::endl; }

	// Train Random Forest classifier
	rtree->train(data);
	std::cout << "Training completed" << std::endl;

	// Store trained model (this is the input xml for the test phase)
	std::string filename_model = "Trained model.xml";
	rtree->save(filename_model);

	// Calculate and print generilization error
	cv::Mat ResponsesForError;
	float trainingError = rtree->calcError(data, true, ResponsesForError);
	std::cout << "Generilization error is: " << trainingError << std::endl;
}

// Testing phase. Apply the classifier to get predicted matchable keypoints for an image
void App::predictOnImage(const cv::Mat & testImageInput, const cv::String trainedModelXMLFile, const cv::String CSVfileTestImagePredictions, std::vector<cv::KeyPoint> &testKps, cv::Mat &testDesc, cv::Mat &labelsPredicted, const int n_featuresWrtMethod) {

	// TODO-Deactivate image showing to improve performance
	cv::String win3, imageSrcDet, fileSrcPred;
	win3 = "Test (source) image";
	imageSrcDet = "Detected keypoints in test";
	fileSrcPred = "Predicted keypoints in test";

	// Downsample to speed up processing
	cv::Mat testImageInput_down;
	const int IMAGE_DOWNSAMPLE = 1; // downsample the image to speed up processing
	cv::resize(testImageInput, testImageInput_down, testImageInput.size() / IMAGE_DOWNSAMPLE);

	// Clone image
	cv::Mat testImage = testImageInput.clone();

	// Detect and describe SIFT keypoints on image
	matching.detectAndDescribe(testImage, testKps, testDesc, imageSrcDet);

	// Calculate features (variables) according to the selected architecture
	cv::Mat1f featuresTest(testKps.size(), n_featuresWrtMethod, CV_32F);
	if (n_featuresWrtMethod == 8)
	{
		featuresTest = varCalc.constructVariableMat(testImage, testKps, testDesc);
	}
	else if (n_featuresWrtMethod == 128)
	{
		featuresTest = testDesc;
	}

	// Load trained model
	cv::Ptr<cv::ml::RTrees> rtreeLoaded = cv::ml::StatModel::load<cv::ml::RTrees>(trainedModelXMLFile);

	// Predict matchable keypoints for the test image
	rtreeLoaded->predict(featuresTest, labelsPredicted);

	// Deactivate image showing and storage to improve performance
	// Distribution of predictions in the two classes
	std::vector<cv::KeyPoint> predKps;
	for (int i = 0; i < labelsPredicted.rows; i++) {
		if (labelsPredicted.at<float>(i, 0) == 1) {
			predKps.push_back(testKps[i]);
		}
	}

	// Store the predicted labels
	std::string rawnamePredictions = "Results/PredictedLabelsAll/PredictedLabelsAll_" + CSVfileTestImagePredictions;
	filedeal.writeCSV(rawnamePredictions, labelsPredicted);

    // Store the keypoint values for those predicted as matchable
    std::string rawnamePredKps = "Results/PredictedMatchableKeypoints/PredictedMatchableKeypoints_" + CSVfileTestImagePredictions;
	cv::Mat1f predKpsMat(predKps.size(), 6, CV_32F);
	for (int j = 0; j < predKps.size(); j++) {
		predKpsMat.at<float>(j, 0) = predKps[j].pt.x;
		predKpsMat.at<float>(j, 1) = predKps[j].pt.y;
		predKpsMat.at<float>(j, 2) = predKps[j].octave;
		predKpsMat.at<float>(j, 3) = predKps[j].angle;
		predKpsMat.at<float>(j, 4) = predKps[j].size;
		predKpsMat.at<float>(j, 5) = predKps[j].response;
	}
	filedeal.writeCSV(rawnamePredKps, predKpsMat);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Match or no match: Keypoint filtering based on matching probability
///
/// Author: Alexandra Papadaki
/// Support: al.i.papadaki@gmail.com
/// Last update : August 2021
///
/// ************************   NOTES       ***************************************************
/// The present algorithm predicts the matchable keypoints in a single image.
/// To obtain your own training data, train a new model or apply inference (testing),
/// please uncomment the corresponding code parts in the main code, and provide the required variables.
///
/// For more details regarding the application of the present algorithm and especially
/// regarding the obtainment of own train data or use of the provided ones,
/// please follow the guidance in the README.txt file that is attached to the algorithm
///
////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Matching.h"
#include "App.h"

#include<iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <time.h>
#include <glob.h>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {

	// INPUTS *********************************************************************************************
	/// For Training and Testing phase: Amount of features which define the classifier architecture. (8=proposed, 128=Predicting matchability 2014)
	// TODO -  change this to run for diff architectures. Also change the file with the model in the directory
	int n_featuresWrtMethod = 8;

	/// For Τraining phase:
	cv::String dirTrainingImages = "Training images"; // Training images folder directory ("Training images" is the default folder for using the provided training images)
	cv::String dirTrainingData = "Training Data"; // Training data folder directory ("Training Data" is the default folder for the provided training data)

	/// For Testing:
	// XML file containing the trained model ("Trained model.xml" is the default value for the provided trained model)
	cv::String trainedModel = "Trained model.xml";

	// File directory for ground truth features (variables) and labels (If no ground truth available, leave it blank)
	//cv::String testFeaturesLabelsFile = "Ground Truth";
	cv::String testFeaturesLabelsFile = "";

	// Testing images folder directory ("Test Images" is the default folder)
	cv::String TestImagesDirectory = "Test Images";


	// MAIN CODE *******************************************************************************************
	clock_t tStart = clock(); //Start counting total time

	/// Τraining phase
	// Please uncomment the following lines to obtain your own trained model.
//	App::getTrainedData(dirTrainingImages, n_featuresWrtMethod);
//	App::trainForest(dirTrainingData, n_featuresWrtMethod);
	// End of training

	/// Testing phase
	// Please uncomment the following lines to apply testing. Adjust the directory folder of your test images in the "TestImagesDirectory" above.
	// Read testing images from directory
	std::vector<cv::String> fn;
	glob(TestImagesDirectory, fn, true);
	cv::Mat testDescSrc;
	cv::Mat testImageInputSrc, testImageInputTar;
	std::vector<cv::KeyPoint> testKpsSrcCut, testKpsTarCut;
	cv::Mat testDescSrcPred, testDescTarPred;
	int iter = fn.size();

	// Iterate over testing images
	for (double k = 0; k < iter; k++)
	{
		cv::String name;
        std::cout << "change src for image " << fn[k];
        testImageInputSrc = imread(fn[k]);
        name = fn[k];

        if (!testImageInputSrc.data) {
            std::cout << "Input test image not found" << std::endl;
            std::cout << "Press enter to exit" << std::endl;
            std::cin.get();
            exit(-2);
        }

        // Predict matchable keypoints for test image
        cv::Mat labelsPredictedSrc;
        std::vector<cv::KeyPoint> testKpsSrc;

        // Define output CSV file with predictions
        // Cut off extension and "Test Images\"
        size_t lastindex = name.find_last_of(".");
        std::string rawname = name.substr(0, lastindex);
        rawname.erase(0, 12);
        std::string rawnameCopy = rawname;
        const char* CSVfileTestImagePredictions = rawnameCopy.c_str();

        clock_t tStartPre = clock();
        App::predictOnImage(testImageInputSrc, trainedModel, CSVfileTestImagePredictions, testKpsSrc, testDescSrc, labelsPredictedSrc, n_featuresWrtMethod);
        printf("Prediction execution time: %.2fs\n", (double)(clock() - tStartPre) / CLOCKS_PER_SEC);

        // Isolate predicted matchable keypoints and descriptors for image Src
        cv::KeyPoint tempKeypointSrc;
        cv::Mat labelsPredictedSrc_Float;
        labelsPredictedSrc.convertTo(labelsPredictedSrc_Float, CV_32F);
        testDescSrcPred.release();
        testKpsSrcCut.clear();
        for (int i = 0; i < labelsPredictedSrc_Float.rows; i++) {
            if (labelsPredictedSrc_Float.at<float>(i, 0) == 1) {
                testDescSrcPred.push_back(testDescSrc.row(i));
                tempKeypointSrc = testKpsSrc.at(i);
                testKpsSrcCut.push_back(tempKeypointSrc);
            }
        }

//        // Write out for VSFM
//        FILE* file1;
//
//        // Create proper sift file name
//        const char *b = ".sift";
//        std::string c = rawname + b;
//        std::string rawnameSift = c.insert(0, "Sift/");
//        const char *nameSift = rawnameSift.c_str();
//
//        file1 = fopen(nameSift, "w");
//        int n1 = testDescSrcPred.rows;
//        int d1 = testDescSrcPred.cols;
//        fprintf(file1, "%d %d\n", n1, d1);
//        for (int i = 0; i < n1; i++) {
//        	// TODO - maybe first x and then y
//        	fprintf(file1, "%f %f %f %f", testKpsSrcCut[i].pt.y, testKpsSrcCut[i].pt.x,	testKpsSrcCut[i].octave, testKpsSrcCut[i].angle);
//
//        	for (int j = 0; j < d1; j++)
//        	{
//        		if (j % 128 == 0)
//        			fprintf(file1, "\n");
//        		fprintf(file1, " %d", (int)(testDescSrcPred.at<float>(i, j)));
//        	}
//        	fprintf(file1, "\n");
//
//        }
//
//        // Write out for ColMap
//        FILE* file2;
//
//        // Create proper sift file name
//        const char *b2 = ".jpg.txt";
//        std::string c2 = rawname + b2;
//        std::string rawnameSift2 = c2.insert(0, "Sift/");
//        const char *nameSift2 = rawnameSift2.c_str();
//
//        file2 = fopen(nameSift2, "w");
//        int n2 = testDescSrcPred.rows;
//        int d2 = testDescSrcPred.cols;
//        fprintf(file2, "%d %d\n", n2, d2);
//        for (int i = 0; i < n2; i++) {
//        	// first y and then x
//        	fprintf(file2, "%f %f %f %f", testKpsSrcCut[i].pt.x, testKpsSrcCut[i].pt.y, testKpsSrcCut[i].octave, testKpsSrcCut[i].angle);
//
//        	for (int j = 0; j < d2; j++)
//        	{
//        		//if (j % 128 == 0)
//        		//	fprintf(file1, "\n");
//        		fprintf(file2, " %d", (int)(testDescSrcPred.at<float>(i, j)));
//        	}
//        	fprintf(file2, "\n");
//
//        }

        std::cout << "Predicted matchable keypoints in test (source) image = " << testDescSrcPred.rows << " (out of " << testKpsSrc.size() << ")" << std::endl;
	}//End of testing

	printf("Total time: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	std::cout << "Press enter to exit" << std::endl;
	cv::waitKey(0);
}

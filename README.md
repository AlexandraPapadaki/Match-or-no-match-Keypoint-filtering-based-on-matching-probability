# Match or no match: Keypoint filtering based on matching probability

Requirements
------------------------
The implemented algorithm predicts matchable keypoints. The introduced algorithm was implemented in C++. It was tested on Windows and Linux OS.

- Additional requirements : OpenCV 3.4.x


Contained source files 
------------------------
	-main.cpp : load data, and call functions for applying pre-training, training and/or testing, according to the needs of the current application.  
	-App.cpp : basic functions for the pre-training, training and testing phase.
	-Matching.cpp : detect and describe keypoints on image, obtain reliable matches for stereo-pairs
	-VariableCalculation.cpp : calculate variables for keypoints and construct the feature (variable) matrix of the samples.
	-FileHandling.cpp : write CSV files and give them proper names


Installation
------------------------
1) Clone code from git repository
2) According to your compiler, create a project and link opencv 3.4.x to it. 
3) Add the downloaded source files (.cpp, .h) to your project.
4) Run training or inference/testing.

Run training (obtain your own model using your own data) 
------------------------
1) Create the following EMPTY folders in your output/project directory:
  	- "Training Images": stereopairs given as pairs in a row. Images can be .jpg, .png, .tiff. (input for pre-training)
  	- "Training Data": one file per image with the variables x, y coordinates, octave, angle, size, response, green color value and dominant orientations of the keypoints in the first 8 columns. The last column contains the label of the keypoint. Each row represents a sample keypoint from the BALANCED data. Columns are seperated by comma. (output from pre-training, input for training)
  	- "Ground Truth": one file per image with the variables x, y coordinates, octave, angle, size, response, green color value and dominant orientations of the keypoints in the first 8 columns. The last column contains the label of the keypoint. Each row represents a sample keypoint from the UNBALANCED, initial data. Columns are seperated by comma. (output from pre-training, input for testing)
2) Provide input: Copy your training images in the "Training Images"
3) Leave uncommented only the training phase in the main.cpp: 
  	- App::getTrainedData(dirTrainingImages, n_featuresWrtMethod);
  	- App::trainForest(dirTrainingData, n_featuresWrtMethod);
4) Compile and run it.
5) Find your output trained model in the "Trained model.xml" file. 

Run inference/testing 
------------------------
1) Provide the "Trained model.xml" in the output/project directory, either using the pretained/downloaded one or your own model.
2) If not created already, create the following EMPTY folders in your output/project directory:
  	- "Ground Truth": Empty if using the provided pretrained model. If training yor own model, then it containes one file per image with the variables x, y coordinates, octave, angle, size, response, green color value and dominant orientations of the keypoints in the first 8 columns. The last column contains the label of the keypoint. Each row represents a sample keypoint from the UNBALANCED, initial data. Columns are seperated by comma. (output from pre-training, input for testing)
  	- "Test Images": Test images in .jpg, .png, .tiff. format (input for testing)
  	- "Results" with subfolders "PredictedLabelsAll" and "PredictedMatchableKeypoints": will containg all the testing/inference results. 
3) Provide input: Copy your testing images in the "Test Images" folder
4) Leave uncommented only the testing phase in the main.cpp and run it
5) Find final predictions under the "Results" folder. "PredictedLabelsAll" folder provides a file for each image, containing the predicted labels for all the detected keypoints. "PredictedMatchableKeypoints" provides a file for each image, containing the keypoints (coordinates, octave, angle, size and response) that were predicted as matchable.

Citation
------------------------
If you use this code for your research, please cite the following paper:

	@inproceedings{papadaki2020match,
	title={Match or no match: Keypoint filtering based on matching probability},
	author={Papadaki, Alexandra I and Hansch, Ronny},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
	pages={1014--1015},
	year={2020}
	}

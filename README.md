# Car Crash Detection using 3D Convolutional Neural Networks
# By: Yasmine Mnafki

This repository contains code for a car crash detector model using a 3D Convolutional Neural Network (3D CNN). The model processes video frames to identify instances of car crashes. This work is part of the  requirements to be fulfilled for BS in Computer Science at the Lebanese American University.

## Requirements

The project requires the following libraries:
- os
- cv2
- glob
- numpy
- tensorflow
- keras
- sklearn

## Dataset
The dataset used for training is available using this [Link](https://drive.google.com/drive/folders/1EkpE0M8NFVRQ0fHHH0bmpDIunhUaC96i?usp=share_linksssss)

## Project Structure

The project contains the following key elements:

- Video preprocessing: The code includes a method for loading and preprocessing video data. It resizes each frame and normalizes pixel values. 
- Custom data generator: There is a custom generator that yields batches of processed videos and corresponding labels.
- Model definition: The model is a 3D CNN with three convolutional layers, each followed by a max pooling layer, and two fully connected layers at the end.
- Model evaluation: After training, the code evaluates the model using different metrics like accuracy, F1-score, precision, recall, and AUC-ROC. 

## Running the Code

1. Ensure you have all the required libraries installed.
2. Prepare your dataset. The code expects video files to be organized in a specific structure in a Google Drive directory. There should be a training and testing directory, each with 'accident' and 'normal' subdirectories containing the corresponding videos.
3. Run the code. It will mount your Google Drive, load and preprocess the data, define the model, and begin the training process. After training, the model will be evaluated on the testing data.
4. At the end, the trained model will be saved as 'crash_detector_experiment3.h5'. You can load and use this model for further testing or deployment.
Note: The path to the dataset might need to be changed depending on the structure of your Google Drive.



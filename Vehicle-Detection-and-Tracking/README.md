# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, our goal is to write a software pipeline to detect vehicles in a video (started with the test_video.mp4 and later implemented on full project_video.mp4).

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, we can also apply a color transform and append binned color features, as well as histograms of color, to our HOG feature vector. 
* Note: for those first two steps we normalized our features and randomize a selection for training and testing.
* Implemented a sliding-window technique and use our trained classifier to search for vehicles in images.
* Ran our pipeline on a video stream (started with the test_video.mp4 and later implemented on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.  


# Face Mask Detection Using YOLOv10s
## Project Overview
This project focuses on detecting the status of face mask usage in real-time using the YOLOv10s model. The system classifies individuals into three categories:

* With Mask – Properly wearing a face mask.

* Without Mask – No face mask detected.

* Mask Worn Incorrectly – The mask is not worn correctly (e.g., below the nose).
## Introduction
### Input:
* Live Video Stream (via webcam)

* Image Files (JPG, PNG, etc.)

### Output:
* Annotated Video Stream (with bounding boxes and labels: "With Mask", "Without Mask", "Incorrect Mask")

* Detection Confidence Score

* Processed images/videos with bounding boxes
## Introducing the dataset
The dataset used is “Face Mask Detection-2 Computer Vision
Project”, which consists of 4,976 images collected from Roboflow. The dataset includes 3 labels:
“Incorrect_Mask”, “With_Mask” and “Without_Mask”, each label is clearly annotated
. The data is divided into three sets: train, validation and test, with the number of images being 3,790
for the training set, 732 for the validation set and 454 for the test set, respectively.

  👉 Link Dataset: https://universe.roboflow.com/ammar-workspace/face-mask-detection-2-oezvq/dataset/4/download
## Predict in real time
This video uses a laptop webcam to predict whether a person is wearing a mask

![Demo](demo/mask_detection.gif)
## Predict in images
The predicted image will display the corresponding prediction results. Users just need to upload the image, and the model will classify the mask-wearing status of the person in the image (correct, incorrect, or no mask).

![Example Prediction](demo/pre1.jpg)

![Example Prediction](demo/pre2.jpg)
## Installation & Setup
### Clone the Repository
    git clone https://github.com/danghohai2004/FaceMaskDetection-YOLOv10s.git
    cd FaceMaskDetection-YOLOv10s
### Install Dependencies (Python 3.9.12)
    pip install -r requirements.txt
### Run Mask Detection code in real time
    python mask_detection.py
# Thank you for watching

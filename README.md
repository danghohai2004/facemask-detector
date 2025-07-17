# 😷 Face Mask Detection with YOLOv10s

## 🚀 Project Overview
A lightweight, real-time system for detecting mask usage status using the YOLOv10s model. It classifies individuals into three categories:

- 🟢 With Mask — correctly worn mask.
- 🔵 Without Mask — no mask detected.
- 🔴 Mask Worn Incorrectly — mask not covering nose/mouth properly.

## 🎯 Goals
- Real-time inference via webcam or image input.
- Confidence scores for each prediction.
- Annotated visual output for both images and video streams.
- Status counter for total predictions by category.

---

## 🧠 Dataset

**Source: [Face Mask Detection-2 (Roboflow)](https://universe.roboflow.com/ammar-workspace/face-mask-detection-2-oezvq/dataset/4/download)**  
- Total images: 4,976  
- Categories: `With_Mask`, `Without_Mask`, `Incorrect_Mask`  
- Splits:  
  - Training: 3,790  
  - Validation: 732  
  - Testing: 454  

---

## ⏱️ Real-Time Webcam Demo
This video uses a laptop webcam to predict whether a person is wearing a mask.

![Demo](demo/mask_detection.gif)

---

## 🖼️ Predict on Static Images
The predicted image will display the corresponding prediction results. Users just need to upload the image, and the model will classify the mask-wearing status of the person in the image (correct, incorrect, or no mask).

![Example Prediction](demo/pre1.jpg)
![Example Prediction](demo/pre2.jpg)

---

## 📂 File Structure
- [model_training.ipynb](./model_training.ipynb)  
- [best.pt](./best.pt)  
- [mask_detection.py](./mask_detection.py)  
- [demo](./demo)  
  - [mask_detection.gif](./demo/mask_detection.gif)  
  - [pre1.jpg](./demo/pre1.jpg)  
  - [pre2.jpg](./demo/pre2.jpg)  
- [requirements.txt](./requirements.txt)
- [README.md](./README.md)

---

## 🔧 Installation & Setup
### Clone the Repository
    git clone https://github.com/danghohai2004/FaceMaskDetection-YOLOv10s.git
    cd FaceMaskDetection-YOLOv10s
### Install Dependencies (Python 3.9.12)
    pip install -r requirements.txt
### Run Real-Time Detection
    python mask_detection.py

---

## 📜 License
This project is licensed under the MIT License. Feel free to use, modify, and distribute.

---

## 📬 Contact

- **📧 Email** — [Liên hệ tôi](mailto:haicoin1324@gmail.com?subject=Hỏi về dự án&body=Chào bạn,%0A%0A)
- **🌐 GitHub** — <a href="https://github.com/danghohai2004" target="_blank">[@danghohai2004](https://github.com/danghohai2004)</a>

---

# Thank you for your interest in this project!




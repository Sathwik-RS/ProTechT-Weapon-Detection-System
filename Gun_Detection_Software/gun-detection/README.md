# Weapon Detection System

This project implements a real-time weapon detection system using the YOLOv8 model. The system is designed to detect and track weapons (specifically guns) within video streams or video files, providing a visual indication by highlighting detected weapons in each video frame.

## Model and Dataset

- **Model Used:** YOLOv8, a state-of-the-art object detection model known for its speed and accuracy.
- **Dataset:** The model is trained on a dataset containing 6,000 annotated images focused on hand-held weapons.
  - **Dataset Link:** [Hand Weapon Dataset on Hugging Face](https://huggingface.co/datasets/AbdulHadi806/hand-weapone-dataset)
  - **Dataset Creation Tool:** The dataset was curated and annotated using [Roboflow](https://roboflow.com/), a popular tool for generating high-quality datasets.

## Technology Stack

- **OpenCV:** Utilized for video processing, frame handling, and drawing bounding boxes around detected objects.
- **YOLOv8:** Employed for its efficient object detection capabilities, enabling the system to identify weapons with high accuracy.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/weapon-detection-system.git
cd weapon-detection-system
python main_area.py
```

<img src="https://github.com/user-attachments/assets/ab92f0b2-6056-477a-967f-b74228f3a903" alt="image" width="500"/>


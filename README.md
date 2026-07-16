# Hybrid Deep Learning Approach Using EfficientNetB0, Conv3D, and BiLSTM for Disease-Specific Indian Sign Language Recognition

This repository contains the Python implementation of a hybrid deep learning framework developed for recognizing disease-specific gestures in Indian Sign Language (ISL). The system is designed to support healthcare communication by automatically recognizing medical and symptom-related sign language gestures from video sequences.

## Overview

Communication between healthcare professionals and individuals who are deaf or mute can become challenging when symptom-specific information needs to be conveyed accurately. This project addresses this problem using a video-based deep learning architecture specifically designed for disease-specific Indian Sign Language recognition.

The proposed architecture combines:

* **Conv3D** for spatiotemporal feature extraction
* **TimeDistributed EfficientNetB0** for deep frame-level spatial feature extraction
* **Bidirectional LSTM (BiLSTM)** for temporal sequence modeling
* **Temporal Attention Mechanism** for emphasizing informative temporal features
* **Softmax Classification** for recognizing 48 disease-specific gesture classes

The proposed framework achieved a validation accuracy of **96.55%** on the curated disease-specific ISL dataset used in this study.

---

## Repository Structure

```text id="27g7p9"
Disease-Specific-Indian-Sign-Language-Recognition/
│
├── disease_specific_isl_recognition.py
├── README.md
├── requirements.txt
└── .gitignore
```

### File Description

* `disease_specific_isl_recognition.py` — Contains the complete Python implementation, including data preprocessing, model architecture, training, and evaluation.
* `README.md` — Contains complete documentation about the project.
* `requirements.txt` — Contains the Python dependencies required to run the project.
* `.gitignore` — Prevents unnecessary files, datasets, model files, and temporary files from being tracked by Git.

---

## Model Architecture

The overall deep learning pipeline follows the structure:

```text id="w9kht2"
Input Video Sequence
        │
        ▼
Video Preprocessing
        │
        ▼
Conv3D
Spatiotemporal Feature Extraction
        │
        ▼
TimeDistributed EfficientNetB0
Spatial Feature Extraction
        │
        ▼
Global Average Pooling
        │
        ▼
Bidirectional LSTM
Temporal Feature Learning
        │
        ▼
Temporal Attention Mechanism
        │
        ▼
Dense Layer + Dropout
        │
        ▼
Softmax Classification
        │
        ▼
48 Disease-Specific ISL Classes
```

The architecture combines spatial and temporal feature learning to capture both the visual characteristics and motion patterns associated with medical sign-language gestures.

---

## Dataset

A dedicated video dataset was used for disease-specific Indian Sign Language recognition.

### Dataset Statistics

| Property            | Details                               |
| ------------------- | ------------------------------------- |
| Total Videos        | 6,980                                 |
| Number of Classes   | 48                                    |
| Data Type           | Video sequences                       |
| Domain              | Disease-specific Indian Sign Language |
| Input Resolution    | 64 × 64 RGB                           |
| Classification Type | Multi-class gesture recognition       |

The dataset contains gestures representing medical symptoms and conditions used in healthcare communication.

Example gesture categories include:

* Fever
* Cough
* Chest Pain
* Headache
* Vomiting
* Nausea
* Other disease-specific and symptom-related gestures

The complete dataset is **not included in this repository** because of its large storage size and data-sharing considerations.

Researchers interested in reproducing the work should use an appropriately structured disease-specific ISL video dataset and follow the preprocessing pipeline implemented in the Python source code.

---

## Dataset Structure

The dataset used during model development follows a class-based directory organization similar to:

```text id="c7z9uq"
dataset/
│
├── class_01/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
│
├── class_02/
│   ├── video_001.mp4
│   └── ...
│
└── class_48/
    ├── video_001.mp4
    └── ...
```

The complete dataset is intentionally excluded from the GitHub repository.

---

## Data Preprocessing

The preprocessing pipeline includes:

1. Reading video sequences using OpenCV.
2. Extracting a fixed number of frames from each video.
3. Resizing frames to **64 × 64 pixels**.
4. Converting frames from BGR to RGB.
5. Normalizing pixel values.
6. Applying temporal padding or truncation when required.
7. Applying data augmentation techniques such as horizontal flipping and brightness adjustment.
8. Computing class weights to reduce the effect of class imbalance.

All preprocessing operations required for model training are implemented directly in the `disease_specific_isl_recognition.py` Python source file.

---

## Training Configuration

The model was trained using the following general configuration:

| Parameter                | Configuration                   |
| ------------------------ | ------------------------------- |
| Optimizer                | Adam                            |
| Loss Function            | Sparse Categorical Crossentropy |
| Initial Learning Rate    | 0.001                           |
| Learning Rate Strategy   | Cosine Decay                    |
| Batch Size               | 16                              |
| Maximum Epochs           | 30                              |
| Number of Classes        | 48                              |
| Input Resolution         | 64 × 64                         |
| Class Imbalance Handling | Class Weighting                 |

Training callbacks and regularization strategies were used to improve convergence and reduce overfitting.

---

## Results

The proposed hybrid architecture achieved:

### **96.55% Validation Accuracy**

The study compared the proposed architecture with several alternative deep learning approaches.

| Model                                           |   Accuracy |
| ----------------------------------------------- | ---------: |
| CNN + BiLSTM                                    |     91.45% |
| 3D CNN + LSTM                                   |     95.79% |
| CNN + LSTM                                      |     95.78% |
| TimeDistributed Attention CNN                   |     95.99% |
| **Conv3D + EfficientNetB0 + BiLSTM (Proposed)** | **96.55%** |
| MobileNetV2 + BiLSTM                            |     87.05% |

The results demonstrate the effectiveness of combining spatiotemporal feature extraction, deep spatial representations, and bidirectional temporal modeling for disease-specific ISL recognition.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* scikit-learn
* Matplotlib
* EfficientNetB0
* Conv3D
* Bidirectional LSTM
* Temporal Attention Mechanism

---

## Installation

### 1. Clone the Repository

```bash id="gwwnct"
git clone ( https://github.com/maheshdattatreya24/Disease-Specific-Indian-Sign-Language-Recognition )
cd Disease-Specific-Indian-Sign-Language-Recognition
```

### 2. Install the Required Dependencies

```bash id="z50ptq"
pip install -r requirements.txt
```

---

## Running the Project

### 1. Configure the Dataset Path

Open the following Python source file:

```text id="ztkxyh"
disease_specific_isl_recognition.py
```

Locate the dataset path configuration and update it according to the location of the dataset on your system.

For example:

```python id="scvifv"
DATASET_PATH = "path/to/your/dataset"
```

Use the actual variable name present in the source code if it differs from the example above.

### 2. Run the Python Script

Open a terminal or command prompt inside the project directory and execute:

```bash id="4pnwzr"
python disease_specific_isl_recognition.py
```

The Python script contains the complete implementation of the project, including:

* Dataset loading
* Video preprocessing
* Data augmentation
* Model architecture construction
* Conv3D-based spatiotemporal feature extraction
* EfficientNetB0-based spatial feature extraction
* BiLSTM-based temporal sequence learning
* Temporal attention
* Model training
* Model evaluation
* Result visualization

> **Note:** The complete dataset is not included in this repository because of its large size. Users must obtain access to an appropriate dataset and configure the local dataset path in the Python source file before running the training pipeline.

---

## Requirements

The project was developed using Python and deep learning/computer vision libraries.

Install all required packages using:

```bash id="h3f4e7"
pip install -r requirements.txt
```

The primary dependencies include:

```text id="y2j1ha"
tensorflow==2.9.0
opencv-python
numpy
scikit-learn
matplotlib
```

Additional dependencies may be required depending on the exact imports used in `disease_specific_isl_recognition.py`.

---

## Research Paper

This repository contains the implementation associated with the research paper:

**"Hybrid Deep Learning Approach Using EfficientNetB0, Conv3D, and BiLSTM for Disease-Specific Indian Sign Language Recognition"**

Published in:

**2025 International Conference on Emerging Technologies in Electronics and Green Energy (ICETEG)**

**DOI:** `10.1109/ICETEG66194.2025.11473053`

### Authors

* Tennety Mahesh Dattatreya
* Bhumika Karsh
* Ram Kumar Karsh

---

## Citation

If you use this project, implementation, or methodology in academic research, please cite the associated research paper.

The official BibTeX citation can be obtained from the IEEE Xplore publication page using the DOI provided above.

---

## Future Work

Potential future extensions include:

* Expanding the dataset with more diverse signers and regional ISL variations
* Incorporating facial expressions and body-pose information
* Exploring transformer-based architectures
* Improving real-time inference performance
* Developing a complete real-time GUI-based healthcare communication system

---

## Important Note

The source code is provided as a single Python file, `disease_specific_isl_recognition.py`, containing the complete implementation of the research workflow.

The dataset is not hosted in this repository because of its large size and applicable data-sharing considerations.

Users who wish to execute the code must configure the dataset path in the Python source file and ensure that the required dependencies are installed.

---

## Disclaimer

This project was developed for research and assistive-technology purposes. The system is not intended to replace professional medical diagnosis or clinical decision-making.

# 🔍 Steel Surface Defect Detection using YOLOv11 & Streamlit

An end-to-end Computer Vision project for automated steel surface defect detection. This project utilizes a fine-tuned **YOLOv11** model trained on Kaggle and deployed as an interactive web application built with **Streamlit**.

---

## 📌 Project Overview

Steel surface defects can significantly compromise industrial product quality. This application automates the detection of **6 common steel defect types**:
1. **Crazing**
2. **Inclusion**
3. **Patches**
4. **Pitted Surface**
5. **Rolled-in Scale**
6. **Scratches**

The model was optimized with an operational confidence threshold of **`conf=0.10`**, effectively balancing high detection recall with minimal false alarm rates for deployment.

---

## 📊 Model Performance & Results

### 1. Test Set Evaluation Metrics

| Metric | Baseline (`conf=0.001`) | Optimal Threshold (`conf=0.10`) |
| :--- | :---: | :---: |
| **Precision** | **70.32%** | **70.32%** |
| **Recall** | **66.49%** | **66.49%** |
| **mAP50** | 73.36% | **68.89%** |
| **mAP50-95** | 41.46% | **38.92%** |

> **Key Takeaway**: Operating at `conf=0.10` retains identical Precision and Recall performance while successfully eliminating background noise and false detections in production.

### 2. Test Set Detection Success Rates

| Defect Class | Total Samples | Detected | Missed | Success Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Inclusion** | 39 | 39 | 0 | **100.00%** |
| **Patches** | 42 | 42 | 0 | **100.00%** |
| **Pitted Surface** | 52 | 52 | 0 | **100.00%** |
| **Rolled-in Scale** | 47 | 47 | 0 | **100.00%** |
| **Scratches** | 45 | 45 | 0 | **100.00%** |
| **Crazing** | 45 | 44 | 1 | **97.78%** |

---

## 📂 Repository Structure

```text
steel-defect-detection-yolov8/
├── models/
│   └── best.pt               # Fine-tuned YOLOv8 model weights
├── notebooks/
│   └── steel_defect_detection.ipynb  # Full Kaggle training & evaluation notebook
├── app.py                    # Main Streamlit web application script
├── requirements.txt          # Python project dependencies
└── README.md                 # Project documentation
```
## ⚙️ Installation & Local Setup
Follow these steps to run the Streamlit application on your local machine:
### 1. Clone the Repository
```Bash
git clone https://github.com/sayidmufaqih/steel-defect-detection-yolov11.git
cd steel-defect-detection-yolov11
```
### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```Bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```Bash
pip install -r requirements.txt
```
### 4. Run the Streamlit Application
```Bash
streamlit run app.py
After executing the command, open your browser at http://localhost:8501.
```
## 🚀 How to Use the Web App
1. Launch the Streamlit application.
2. Upload a steel surface image (.jpg, .jpeg, or .png).
3. Click the Detect Defects button.
4. View the bounding box predictions and detected defect categories directly on the screen!

## 🛠️ Tech Stack & Libraries
- Computer Vision: Ultralytics YOLOv8, OpenCV, PIL
- Deep Learning Framework: PyTorch
- Data Processing & Analytics: Pandas, Matplotlib, Scikit-Learn
- Web Deployment: Streamlit

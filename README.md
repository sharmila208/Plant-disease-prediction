# 🌿 Plant Disease Prediction Web App

A Deep Learning-based web application that predicts plant leaf diseases using a trained CNN model. Built using TensorFlow and Streamlit, this project is designed with a greenery-themed interface and provides disease-specific precautions.

---

## 🧠 Features
- 📸 Upload leaf images or use your camera
- 🧠 Predicts the disease using a CNN model
- 🩺 Displays disease name and suggested precautions
- 🌐 Future scope: Multi-language support & PDF report export
- 🎨 Greenery-themed UI with nature vibes

---

## 📁 Files in This Repository

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web app code |
| `Modified_Plant_Disease_Prediction.ipynb` | Final training notebook for the CNN model |
| `class_indices.json` | Maps predicted class indices to disease names |
| `plantvillage-dataset-metadata.json` | Metadata about the dataset used |

---

## ⚙️ How to Run Locally

Make sure you have Python installed.

```bash
git clone https://github.com/yourusername/plant-disease-prediction.git
cd plant-disease-prediction
pip install streamlit tensorflow
streamlit run app.py


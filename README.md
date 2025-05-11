# Brain Tumor Classification using Deep Learning

## 📌 Project Overview
This project focuses on developing a deep learning-based system to classify brain MRI images into four categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

Leveraging **Convolutional Neural Networks (CNNs)** and **Transfer Learning (EfficientNetB0)**, the system assists in early and accurate brain tumor detection to support clinical diagnosis.

---

## 📂 Dataset
- **Source:** [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Categories:**
  - `glioma_tumor`
  - `meningioma_tumor`
  - `pituitary_tumor`
  - `no_tumor`
- **Preprocessing:**
  - Image resizing to `48x48`
  - Normalization
  - Label encoding

---

## 🧠 Models Implemented
1. **Multilayer Perceptron (MLPClassifier)** - Scikit-learn baseline model.
2. **EfficientNetB0** - Transfer learning using Keras' pretrained model.
3. **Custom CNN** - Built from scratch using TensorFlow/Keras.

Each model is evaluated on:
- Accuracy
- Confusion Matrix
- Loss & Accuracy Curves

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:**
  - `TensorFlow`, `Keras`
  - `OpenCV`, `Matplotlib`, `NumPy`, `Pandas`
  - `Scikit-learn`
- **Platform:** Google Colab (with GPU support)

---

## 🚀 How to Run

### Setup:
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

### Dependencies:
```bash
pip install tensorflow opencv-python matplotlib scikit-learn pandas
```

### Train & Evaluate:
Run the notebook or Python script in Google Colab or locally. Ensure your dataset is structured by class folders under `Training` and `Testing`.

---

## 📊 Results Summary

| Model         | Accuracy (Test Set) |
|---------------|---------------------|
| MLPClassifier | ~85%                |
| EfficientNetB0| ~95%+               |
| CNN (Custom)  | ~93%+               |

EfficientNetB0 with transfer learning provided the best overall performance and generalization.

---

## 🎯 Project Goals
- Improve diagnostic accuracy in neuro-oncology.
- Reduce manual interpretation burden on radiologists.
- Deliver a scalable and adaptable ML model for clinical settings.

---

## 🔮 Future Enhancements
- Add support for multimodal input (genomic + clinical).
- Implement explainable AI (Grad-CAM, SHAP).
- Deploy via a Flask web app or Streamlit for real-time inference.

---

## 🤝 Acknowledgements
- Supervisor: *Mr. Azhar Mahmood*
- Dataset: [Kaggle Contributors](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

---

## 📃 License
This project is licensed under the [MIT License](LICENSE).

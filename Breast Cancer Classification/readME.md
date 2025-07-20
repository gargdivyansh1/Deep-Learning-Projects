# 🧬 Breast Cancer Classification using Deep Learning

This project predicts whether a tumor is **Malignant** or **Benign** using structured numerical data from the **Breast Cancer Wisconsin (Diagnostic) dataset**. The model is built using TensorFlow/Keras and deployed using Streamlit for interactive prediction.

## 📁 Dataset

- Source: [sklearn.datasets.load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- Features: 30 real-valued features computed from digitized images of breast mass.
- Target:
  - `0` - Malignant
  - `1` - Benign

## 🧠 Model Architecture

- Preprocessing:
  - Standardized features using `StandardScaler`
- Neural Network:
  - Input layer (30 features)
  - Dense Layer 1: 32 neurons, ReLU activation
  - Dense Layer 2: 16 neurons, ReLU activation
  - Output Layer: 1 neuron, Sigmoid activation
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

## 🛠️ Dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

## Requirements
- numpy
- pandas
- scikit-learn
- tensorflow
- streamlit

## Testing Example
```bash
[14.1, 20.0, 92.0, 600.0, 0.1, 0.13, 0.2, 0.09, 0.2, 0.07, 0.3, 1.5, 2.0, 25.0, 0.007, 0.04, 0.05, 0.015, 0.02, 0.006, 16.0, 30.0, 110.0, 800.0, 0.15, 0.25, 0.3, 0.1, 0.3, 0.08]
```

- Benign tumor with 96.3% confidence.

## 👤 Author

**Divyansh Garg**    
📧 Email: [divyanshgarg515@gmail.com](mailto:divyanshgarg515@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/divyansh-garg515/)  
💻 [GitHub](https://github.com/gargdivyansh1)


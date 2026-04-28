# medical-triage
deep learning project
# 🏥 AI-Based Medical Triage System using Deep Learning

An intelligent system that automatically predicts the **appropriate medical specialty** from patient symptom descriptions using **Natural Language Processing (NLP)** and **Deep Learning models (BiLSTM & BioBERT)**.

---

## 🚀 Overview

In modern healthcare systems, patient triage is a critical yet time-consuming task. This project aims to **automate the triage process** by analyzing free-text symptom descriptions and classifying them into the correct medical department.

👉 Example:

```
Input: "Chest pain radiating to left arm with shortness of breath"
Output: Cardiovascular / Pulmonary
```

---

## 🎯 Objectives

* Automate medical triage using AI
* Reduce workload on healthcare professionals
* Improve decision-making accuracy
* Compare performance of deep learning models

---

## 🧠 Models Used

### 🔹 BiLSTM (Bidirectional LSTM)

* Captures sequential dependencies in text
* Uses word embeddings and memory cells
* Baseline deep learning model

### 🔹 BioBERT (Transformer-Based Model)

* Pretrained on biomedical text
* Uses attention mechanism
* Captures contextual meaning of medical terms
* Achieved best performance

---

## 🛠️ Tech Stack

* Python 🐍
* PyTorch 🔥
* HuggingFace Transformers 🤗
* Scikit-learn
* NumPy, Pandas
* Matplotlib, Seaborn

---

## 📂 Dataset

* Medical Transcriptions Dataset
* Medical Speech & Intent Dataset

Features:

* Patient symptom descriptions (text)
* Corresponding medical specialties

---

## ⚙️ Workflow

```
Data Collection → Preprocessing → Model Training → Evaluation → Prediction
```

### 🔹 Preprocessing Steps

* Lowercasing
* Removing special characters
* Tokenization
* Padding / Encoding
* Label encoding

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

---


### 🔥 Key Insights

* BioBERT outperformed all models
* Deep learning models handle medical text better
* Contextual understanding improves classification

---

## 📉 Visualizations

* Accuracy vs F1 Score Comparison
* Loss Curves (Training vs Validation)
* Confusion Matrix
* Per-Class Performance

---

## ⚠️ Challenges

* Limited labeled medical data
* Complex medical terminology
* High computational cost (BERT models)

---

## 🚀 Future Scope

* Real-time hospital deployment
* Multilingual support
* Integration with hospital systems
* Explainable AI (XAI)

---

## 🧪 Installation

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn
```

---

## ▶️ Run the Project

```bash
python main.py
```

*or use Jupyter Notebook / Google Colab*

---

## 📌 Project Structure

```
├── data/
├── models/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── lstm_model.py
│   ├── bert_model.py
│   └── train.py
├── results/
├── README.md
```

---



---

## 📜 License

This project is for academic and research purposes.

---

## ⭐ Acknowledgements

* HuggingFace Transformers
* PyTorch
* Kaggle Datasets

---

## 💡 Final Note

This project demonstrates how **AI can assist healthcare systems** by improving efficiency, accuracy, and decision-making in patient triage.

---

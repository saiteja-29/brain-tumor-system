# 🧠 Brain Tumor Detection System

An end-to-end deep learning system for detecting brain tumors from MRI images, featuring real-time inference, explainable AI (Grad-CAM), and cloud deployment.

---

## 🚀 Features

* 🧠 MRI image classification using **EfficientNet (Transfer Learning)**
* ⚡ Real-time prediction using **FastAPI**
* 🔍 **Grad-CAM visualization** for model interpretability
* 🌐 Interactive **frontend UI** for image upload and results
* 🐳 Containerized using **Docker** for reproducibility
* ☁️ Deployed on cloud (Render)

---

## 🛠️ Tech Stack

* **Machine Learning:** TensorFlow, Keras
* **Backend:** FastAPI
* **Frontend:** HTML, CSS, JavaScript
* **Computer Vision:** OpenCV
* **Deployment:** Docker, Render

---

## 📊 Model Performance

The model is evaluated using multiple metrics suitable for medical diagnosis:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

> ⚠️ In medical applications, **Recall and F1-score** are prioritized to minimize false negatives (missing tumors).

---

## 🔬 Explainability (Grad-CAM)

Grad-CAM is used to highlight regions in MRI images that influence the model’s predictions.

* 🔴 Red regions → High importance
* 🔵 Blue regions → Low importance

This improves trust and interpretability of the model.

---

## 🖥️ Demo

👉 **Live Demo:** *[[Click here to view the website](https://brain-tumor-system-q8ze.onrender.com/)]*

---

## 📂 Project Structure

```
brain-tumor-system/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   └── model.h5
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│
├── api/
│   ├── main.py
│   └── frontend/
│       ├── index.html
│       ├── style.css
│       ├── script.js
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd brain-tumor-system
mkdir data
cd data
mkdir raw
mkdir processed
cd ..

```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Run the API

```
uvicorn api.main:app --reload
```

---

### 4️⃣ Open in browser

```
http://127.0.0.1:8000
```

---

## 📌 API Endpoint

### 🔹 POST `/predict`

Upload an MRI image and get:

* Prediction (tumor / no_tumor)
* Confidence score
* Grad-CAM heatmap (base64 image)

---

## 🧠 Key Highlights

* Built a **production-ready ML system**, not just a model
* Integrated **explainable AI (Grad-CAM)** for medical interpretability
* Designed **end-to-end pipeline** from data → model → API → UI → deployment
* Solved real-world deployment issues (Docker, OpenCV dependencies, API integration)

---

## 🚀 Future Improvements

* Add user authentication
* Improve model with larger dataset
* Deploy frontend separately (React)
* Add monitoring & logging

---

## 🤝 Contributing

Feel free to fork the repo and submit pull requests!

---

## 📜 License

This project is open-source and available under the MIT License.

---

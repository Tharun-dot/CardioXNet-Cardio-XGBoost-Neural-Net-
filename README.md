# ❤️ Heart Attack Prediction using Knowledge Distillation

This project predicts the **risk of heart attack** using machine learning and deep learning.
It combines a **Gradient Boosting Classifier (teacher model)** with **Neural Network distillation (student model)** to improve performance and generalization.

---

## 🚀 Features

* 🩺 Predicts **heart attack risk** based on patient health data
* 📊 Uses **Gradient Boosting with GridSearchCV** for optimal accuracy
* 🎓 Implements **Knowledge Distillation**:

  * Teacher: Gradient Boosting Classifier
  * Student: Lightweight Neural Network trained with **soft labels**
* 🔥 Achieves high accuracy (**~98.5%**) with robust evaluation
* ✅ Supports **custom input prediction**

---

## 🛠️ Tech Stack

* **Machine Learning:** scikit-learn (GradientBoosting, GridSearchCV, StandardScaler)
* **Deep Learning:** PyTorch (Neural Network, KL Divergence Loss, Adam Optimizer)
* **Other:** Pandas, NumPy, Joblib / Pickle for model saving

---

## 📂 Project Structure

```
heart-attack-prediction/
│── heart.csv                 # Dataset
│── teacher_student.py        # Training pipeline (teacher + student)
│── model.pkl                 # Saved teacher model
│── scaler.pkl                # Saved feature scaler
│── soft_label.npy            # Soft labels from teacher model
│── README.md                 # Project documentation
```

---

## 📊 Dataset

The dataset (`heart.csv`) contains patient health records with features such as:

* `age`: Age of the patient
* `sex`: Gender (0 = female, 1 = male)
* `trestbps`: Resting blood pressure
* `chol`: Serum cholesterol (mg/dl)
* `thal`: Thalassemia type
* `cp`: Chest pain type
* `exang`: Exercise-induced angina
* `target`: 0 = No risk, 1 = High risk

---

## ⚙️ Model Training Pipeline

1. **Teacher Model (Gradient Boosting Classifier)**

   * Trained with top 7 features
   * Hyperparameter tuning with `GridSearchCV`
   * Best parameters: `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 150}`
   * Accuracy: **98.54%**, AUC: **0.9854**

2. **Knowledge Distillation**

   * Teacher outputs soft labels (`predict_proba`)
   * Student model = small feedforward **Neural Network (PyTorch)**
   * Trained with **hybrid loss**:

     * CrossEntropyLoss (hard labels)
     * KLDivLoss (soft labels from teacher)
   * Achieves stable loss convergence

---

## 📊 Example Results

**Teacher Model Evaluation**

```
Gradient Boosting with Top 7 Features + GridSearchCV
Best Parameters: {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 150}
Accuracy: 98.54%
AUC Score: 0.9854
```

**Classification Report**

```
              precision    recall  f1-score   support
0             0.97        1.00     0.99       102
1             1.00        0.97     0.99       103
Accuracy      0.99
```

**Student Model Training (Example)**

```
Epoch 0, Loss: 6.8483
Epoch 10, Loss: 2.8921
Epoch 20, Loss: 1.0230
Epoch 30, Loss: 0.9255
Epoch 40, Loss: 0.8788
```

**Custom Input Prediction**

```
✅ No Significant Risk of Heart Attack — Keep maintaining a healthy lifestyle.
```

---

## 🧪 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Tharun-dot/heart-attack-prediction.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Train the teacher model & save artifacts:

   ```bash
   python teacher_student.py
   ```
4. Predict with custom input:

   ```python
   custom_input = np.array([[35, 0, 110, 170, 2, 1, 0]])
   ```

   Output:

   ```
   ✅ No Significant Risk of Heart Attack — Keep maintaining a healthy lifestyle.
   ```

---

## 🎯 Future Enhancements

* Add **visual dashboards** (Streamlit) for patient-friendly results
* Use **more features** from dataset for better prediction
* Explore **advanced distillation techniques** (temperature scaling, multi-teacher distillation)

---
* LinkedIn: [Your LinkedIn Profile]
* Portfolio: [Your Portfolio Link]

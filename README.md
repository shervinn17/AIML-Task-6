# 🧠 K-Nearest Neighbors (KNN) Classification

## 📌 Objective
This project aims to **understand and implement K-Nearest Neighbors (KNN)** for solving classification problems using Python and scikit-learn.

---

## 📂 Dataset
The dataset used is a CSV file provided for the task. It contains labeled instances for classification, where:

- **Features**: All columns except the last
- **Target**: Last column (label)

---

## 🛠️ Tools & Libraries
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## 🚀 Project Steps

### 1. Load and Explore Data
- Load dataset using `pandas`
- Inspect shape, types, and missing values

### 2. Preprocess Data
- Normalize features using `StandardScaler`
- Split dataset into training and testing sets (80/20 split)

### 3. Build KNN Classifier
- Use `KNeighborsClassifier` from `sklearn.neighbors`
- Train models with different values of **K** (1 to 10)
- Select the best `K` based on accuracy

### 4. Evaluate Performance
- Evaluate model using:
  - **Accuracy Score**
  - **Confusion Matrix**
- Plot confusion matrix using `ConfusionMatrixDisplay`

### 5. Visualize Decision Boundaries (if 2D)
- Visualize how KNN separates classes when dataset has **2 numerical features**
- Show decision regions with `matplotlib.contourf`

---

## 📈 Output Examples

- Accuracy scores for K=1 to K=10
- Confusion matrix for the best `K`
- Decision boundary plot (for 2-feature datasets)

---

## 📊 Sample Code Used

```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

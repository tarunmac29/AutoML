# 🤖 Smart AutoML

A production-ready, end-to-end AutoML web application built with **Streamlit**, **scikit-learn**, and **SHAP**.

Upload any dataset → Auto-train the best model → Explain predictions → Download model.

---

## 🔥 Features

| Feature | Details |
|---|---|
| **Smart Data Loading** | CSV, Excel (.xlsx/.xls), JSON |
| **Auto Problem Detection** | Classification vs Regression |
| **Intelligent Model Selection** | Small data → Logistic/Linear · Large data → Random Forest |
| **Imbalance Handling** | `class_weight='balanced'` auto-applied |
| **Hyperparameter Tuning** | GridSearchCV with cross-validation |
| **Feature Selection** | SelectKBest (f_classif / f_regression) |
| **Visualizations** | Histograms, Correlation Heatmap, Target Distribution |
| **SHAP Explainability** | Bar plot + Beeswarm summary |
| **Prediction UI** | Dynamic form based on dataset features |
| **Model Download** | Export trained model as `.pkl` |
| **Docker Ready** | Full Dockerfile included |

---

## 🏗️ Project Structure

```
smart_automl/
│
├── app/
│   └── webapp.py              # Main Streamlit application
│
├── src/
│   ├── data_loader.py         # File loading (CSV, Excel, JSON)
│   ├── preprocessing.py       # Cleaning, encoding, analysis
│   ├── feature_selection.py   # SelectKBest feature selection
│   ├── model_trainer.py       # GridSearchCV, model selection, saving
│   ├── explainability.py      # SHAP explainability
│   └── utils.py               # Plots, metrics, serialization
│
├── models/
│   └── model.pkl              # Saved model (auto-generated)
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone or download the project
cd smart_automl

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/webapp.py
```

Open your browser at: **http://localhost:8501**

---

## 🐳 Run with Docker

### Build the image

```bash
docker build -t smart-automl .
```

### Run the container

```bash
docker run -p 8501:8501 smart-automl
```

Open your browser at: **http://localhost:8501**

### Run with volume (to persist saved models)

```bash
docker run -p 8501:8501 -v $(pwd)/models:/app/models smart-automl
```

---

## 📊 How It Works

### 1. Upload Dataset
- Drag & drop a CSV, Excel, or JSON file into the sidebar uploader.

### 2. Select Target Column
- Choose the column you want to predict from the dropdown.

### 3. Click "Run AutoML"
The pipeline automatically:
1. **Analyzes** the dataset (size, problem type, class balance)
2. **Preprocesses** (removes duplicates, fills nulls, encodes categoricals)
3. **Selects features** using statistical scoring (SelectKBest)
4. **Trains multiple models** with GridSearchCV
5. **Picks the best model** by F1 (classification) or R² (regression)
6. **Saves** the model to `models/model.pkl`

### 4. Explore Results
- View model metrics, feature importances, SHAP plots
- Use the prediction form to test with custom inputs
- Download the trained model

---

## 🧠 Model Selection Logic

| Condition | Model Chosen |
|---|---|
| Small dataset + Classification | Logistic Regression → Random Forest |
| Large dataset + Classification | Random Forest → Logistic Regression |
| Small dataset + Regression | Linear Regression → Random Forest |
| Large dataset + Regression | Random Forest → Linear Regression |
| Imbalanced classes | `class_weight='balanced'` applied |

> Best model is selected by highest cross-validated F1 (classification) or R² (regression).

---

## 📦 Tech Stack

- **[Streamlit](https://streamlit.io)** — Web UI
- **[scikit-learn](https://scikit-learn.org)** — ML models, GridSearchCV, preprocessing
- **[SHAP](https://shap.readthedocs.io)** — Model explainability
- **[pandas](https://pandas.pydata.org)** — Data manipulation
- **[seaborn](https://seaborn.pydata.org)** + **[matplotlib](https://matplotlib.org)** — Visualization
- **[joblib](https://joblib.readthedocs.io)** — Model serialization

---

## 🤝 Contributing

PRs welcome! For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT

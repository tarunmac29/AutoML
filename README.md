# 🤖 Smart AutoML

A production-ready, end-to-end AutoML web application built with **Streamlit**, **scikit-learn**, **XGBoost**, and **SHAP**.

Upload any dataset → Auto-engineer features → Train the best model → Explain predictions → Serve via REST API.

---

## 🔥 Features

| Feature | Details |
|---|---|
| **Flexible Data Input** | Upload CSV / Excel / JSON, paste raw CSV text, or load built-in sample datasets |
| **Auto Problem Detection** | Automatically detects Classification vs Regression |
| **Auto Data Profiling** | Self-contained HTML EDA report (stats, histograms, correlation heatmap) |
| **Auto Feature Engineering** | Log transforms, polynomial features, pairwise interaction terms |
| **Feature Selection** | SelectKBest (f_classif / f_regression) with configurable K |
| **Intelligent Model Selection** | Trains Random Forest, Logistic/Linear Regression, and XGBoost; picks the best by CV score |
| **Imbalance Handling** | `class_weight='balanced'` auto-applied for imbalanced classification |
| **Hyperparameter Tuning** | GridSearchCV with configurable cross-validation folds |
| **Advanced Visualizations** | Confusion Matrix, ROC Curve, Precision-Recall Curve, Residual Plots, Model Comparison Dashboard |
| **SHAP Explainability** | Bar plot + Beeswarm summary for any trained model |
| **Interactive Prediction UI** | Dynamic input form generated from dataset features |
| **Model Download** | Export trained model as `.pkl` |
| **FastAPI REST API** | `/predict` (single) and `/predict/batch` (CSV upload) endpoints |
| **Docker Ready** | Full Dockerfile; exposes both Streamlit (8501) and FastAPI (8000) |

---

## 🏗️ Project Structure

```
AutoML/
│
├── app/
│   └── webapp.py              # Main Streamlit application (v2.0)
│
├── src/
│   ├── data_loader.py         # File loading (CSV, Excel, JSON)
│   ├── preprocessing.py       # Cleaning, encoding, train/test split
│   ├── feature_engineering.py # Log transforms, polynomial, interactions
│   ├── feature_selection.py   # SelectKBest feature selection
│   ├── model_trainer.py       # GridSearchCV, model selection, saving
│   ├── explainability.py      # SHAP explainability plots
│   ├── advanced_viz.py        # ROC, Confusion Matrix, Residuals, Comparison
│   ├── profiling.py           # Auto EDA HTML report generator
│   ├── api.py                 # FastAPI REST API server
│   └── utils.py               # Plots, metrics, serialization helpers
│
├── models/
│   ├── model.pkl              # Saved model (auto-generated after training)
│   └── model_meta.json        # Model metadata for API (feature names, classes)
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
# 1. Clone the repository
git clone https://github.com/tarunmac29/AutoML.git
cd AutoML

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app/webapp.py
```

Open your browser at: **http://localhost:8501**

### Run the FastAPI server (optional)

After training a model via the Streamlit UI, start the REST API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: **http://localhost:8000/docs**

---

## 🐳 Run with Docker

### Build the image

```bash
docker build -t smart-automl .
```

### Run the Streamlit app

```bash
docker run -p 8501:8501 smart-automl
```

Open your browser at: **http://localhost:8501**

### Run with volume (to persist saved models)

```bash
docker run -p 8501:8501 -p 8000:8000 -v $(pwd)/models:/app/models smart-automl
```

---

## 📊 How It Works

### 1. Load Data
Choose one of three input modes from the sidebar:
- **Upload file** — drag & drop a CSV, Excel (.xlsx/.xls), or JSON file
- **Paste CSV text** — paste raw CSV content directly into the UI
- **Sample dataset** — load the built-in Iris (classification) or California Housing (regression) dataset

### 2. Configure & Run AutoML
Select the target column, configure optional settings (CV folds, test size, feature engineering), then click **Run AutoML**. The pipeline automatically:

1. **Profiles** the dataset — generates a full EDA HTML report
2. **Preprocesses** — removes duplicates, fills nulls, encodes categoricals
3. **Engineers features** — applies log transforms, polynomial features, and/or interaction terms (configurable)
4. **Selects features** — ranks columns using statistical scoring (SelectKBest)
5. **Trains multiple models** — Random Forest, Logistic/Linear Regression, XGBoost — each with GridSearchCV
6. **Picks the best model** — by F1 score (classification) or R² score (regression)
7. **Saves** the model to `models/model.pkl` and metadata to `models/model_meta.json`

### 3. Explore Results
- **Metrics** — accuracy, F1, precision, recall (classification) or R², RMSE, MAE (regression)
- **Advanced charts** — Confusion Matrix, ROC Curve, Precision-Recall Curve, Residual Plots, Model Comparison Dashboard
- **SHAP plots** — feature importance bar chart and beeswarm summary
- **Prediction form** — test custom inputs interactively
- **Download** — export the trained model as a `.pkl` file

---

## 🌐 REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check — returns model loaded status |
| `GET` | `/info` | Model metadata (name, problem type, feature names) |
| `POST` | `/predict` | Single prediction from JSON feature dict |
| `POST` | `/predict/batch` | Batch predictions from uploaded CSV file |

### Example: single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 1.5, "feature2": 3.2}}'
```

### Example: batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@your_data.csv"
```

---

## 🧠 Model Selection Logic

| Condition | Models Evaluated |
|---|---|
| Classification | Logistic Regression, Random Forest, XGBoost |
| Regression | Linear Regression, Random Forest, XGBoost |
| Imbalanced classes | `class_weight='balanced'` applied automatically |

> The best model is selected by the highest cross-validated F1 score (classification) or R² score (regression).

---

## 🔧 Feature Engineering Options

| Option | What It Does |
|---|---|
| **Log transform** | Applies `log1p` to right-skewed numeric columns (skewness > threshold) |
| **Interaction terms** | Adds pairwise multiplication features for top N numeric columns |
| **Polynomial features** | Generates degree-N polynomial + interaction features via `PolynomialFeatures` |

All options are toggleable from the Streamlit sidebar before training.

---

## 📦 Tech Stack

| Library | Purpose |
|---|---|
| **[Streamlit](https://streamlit.io)** | Web UI |
| **[scikit-learn](https://scikit-learn.org)** | ML models, GridSearchCV, preprocessing, feature selection |
| **[XGBoost](https://xgboost.readthedocs.io)** | Gradient boosting models |
| **[SHAP](https://shap.readthedocs.io)** | Model explainability |
| **[FastAPI](https://fastapi.tiangolo.com)** | REST API server |
| **[pandas](https://pandas.pydata.org)** | Data manipulation |
| **[seaborn](https://seaborn.pydata.org)** + **[matplotlib](https://matplotlib.org)** | Visualization |
| **[joblib](https://joblib.readthedocs.io)** | Model serialization |

---

## 🤝 Contributing

PRs welcome! For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT

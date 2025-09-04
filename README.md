# ML_PROJECT_Layoff_Insight
Layoff Insight: Predicts layoff severity (Low, Medium, High, Unknown) from company data using machine learning. Built with FastAPI, Streamlit, and XGBoost.


# 🚀 Layoff Insight

Predict the severity of corporate layoffs (Low, Medium, High, Unknown) based on company data.  
Built with FastAPI, Streamlit, and XGBoost.

---

## 💡 What does this app do?

Layoff Insight predicts how severe a company’s layoff event is, using:
- Total employees laid off
- Percentage of workforce laid off
- Funds raised by the company
- Industry sector
- Country
- Company growth stage

⚠️ **Note:** It does NOT predict the reasons for layoffs (e.g. AI replacing jobs). It predicts only the severity level.

Example prediction:
> “Medium severity (10-50%) layoff for a Series B AI company in the US that laid off 35% of its staff.”

---

## 🗂️ Project Structure
layoff-project/
│
├── api/ # FastAPI backend
│ ├── main.py
│ └── models/
│
├── frontend/ # Streamlit frontend app
│ ├── app.py
│ ├── pages/
│ └── ...
│
├── model/ # Preprocessing pipeline code
│ └── preprocess_pipeline.py
│
├── models/ # Saved trained model
│ └── layoff_pipeline.joblib
│
├── scripts/ # Training script
│ └── train_model.py
│
├── requirements.txt # Python dependencies
└── README.md


---

## ⚙️ How to Run the Project

Follow these steps!

---

### 1. Clone the repo

```bash
git clone https://github.com/yanivbohbot/layoff-project.git
cd layoff-projec
```
### 2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Train the model 
If you want to retrain the model from scratch:
```
python -m api.scripts.train_model
```
Otherwise, a trained model is already saved at:
```
api/models/layoff_pipeline.joblib
```
### 5. Start the FastAPI backend
```
uvicorn api.main:app --reload
```
It should print:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```
Test your API:

    Go to Swagger UI:
    http://127.0.0.1:8000/docs

    Try the POST /predict endpoint!

### 6. Run the Streamlit frontend
```
cd frontend
streamlit run layoff_app.py
```

### Example API request 🎯
```
Send a JSON like this:
{
    "total_laid_off": 200,
    "perc_laid_off": 35,
    "funds_raised": "$120",
    "industry": "AI",
    "country": "United States",
    "stage": "Series B"
}

{
    "predicted_class": 1,
    "severity_label": "Medium (10-50%)"
}

Send a JSON like this:
{
    "total_laid_off": 15,
    "perc_laid_off": 4.5,
    "funds_raised": "300",
    "industry": "Logistic",
    "country": "United States",
    "stage": "Series A"
}

{
    "predicted_class": 1,
    "severity_label": "Predicted Severity: Low (≤10%)"
}



```

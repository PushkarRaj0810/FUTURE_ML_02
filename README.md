# Support Ticket Classification & Prioritization

A machine learning project that classifies support tickets into categories using **TF-IDF and Logistic Regression** and estimates ticket priority using transparent, rule-based logic. The application is built with Streamlit.

## Features

- **Ticket Classification:** Predicts one of eight categories using a trained machine learning pipeline.
- **Text Processing:** Uses TF-IDF vectorization, lowercasing, English stop-word removal, and unigram/bigram features.
- **Priority Estimation:** Assigns High, Medium, or Low priority based on urgency and impact keywords.
- **Model Evaluation:** Reports accuracy, precision, recall, and F1-score on a stratified test set.
- **Priority Rule Tests:** Includes unit tests for priority assignment.

## Dataset

The project uses `all_tickets_processed_improved_v3.csv`, containing ticket text (`Document`) and category labels (`Topic_group`).

- **Total tickets:** 47,837
- **Categories:** 8
- **Input feature:** Ticket description
- **Target:** Ticket category

Place the dataset in the project root directory before training. The dataset is not included in this repository.

## Model & Evaluation

The classification pipeline consists of:

1. **TF-IDF Vectorizer:** Converts ticket text into numerical features.
2. **Logistic Regression:** Learns to predict ticket categories from labeled examples.

The dataset is split into 80% training data and 20% testing data using stratified sampling.

| Metric | Result |
|---|---:|
| Test accuracy | 85.23% |
| Weighted precision | 86% |
| Weighted recall | 85% |
| Weighted F1-score | 85% |

These results measure **category classification**, not priority prediction.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/PushkarRaj0810/FUTURE_ML_02.git
cd FUTURE_ML_02
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the application

The repository includes the trained model `ticket_model.joblib`.

```powershell
streamlit run app.py
```

### 5. Retrain the model

After placing `all_tickets_processed_improved_v3.csv` in the project root, run:

```powershell
python train_model.py
```

This trains the pipeline, prints the evaluation metrics, and saves the updated model.

### 6. Run priority-rule tests

```powershell
python -m unittest test_priority.py
```

The current priority-rule test suite contains eight tests.

## Priority Estimation

The dataset contains category labels but no priority labels. Therefore, priority is **not predicted by a trained machine learning model**.

Instead, `priority.py` uses manually defined keyword rules to estimate priority:

- **High:** Urgency or impact phrases such as "critical", "outage", or "data loss".
- **Medium:** Issue phrases such as "not working", "failed", or "cannot access".
- **Low:** No recognized High- or Medium-priority signal is detected.

The priority logic also accounts for negated urgency cues. The application displays the detected signal to help users understand the estimate.

These estimates are rule-based and may be incorrect. Review them against your organization's service-level policy before taking action.

## Project Structure

```text
FUTURE_ML_02/
├── app.py
├── train_model.py
├── priority.py
├── test_priority.py
├── ticket_model.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- Classification performance depends on the dataset and the quality of its labels.
- Priority estimates use predefined rules rather than learned priority labels.
- A predicted category or estimated priority is not guaranteed to be correct.

## Technologies

Python · Pandas · Scikit-learn · TF-IDF · Logistic Regression · Streamlit · Joblib · Unittest

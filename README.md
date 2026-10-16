# Support Ticket Classification

This Streamlit app predicts a support-ticket category with a TF-IDF and
Logistic Regression pipeline. It separately estimates priority from explicit
urgency and impact phrases, and shows the signal behind that estimate.

## Run the app

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The repository includes a trained `ticket_model.joblib`. To retrain it from
`all_tickets_processed_improved_v3.csv`:

```powershell
python train_model.py
```

The training script cleans empty ticket text, lowercases and tokenizes text
through scikit-learn's TF-IDF vectorizer, removes English stop words, and
reports accuracy, precision, recall, and F1 on a stratified held-out split.

## Priority estimates

Priority is a transparent rule-based estimate, not a model trained on priority
labels: the supplied dataset contains ticket text and category labels only.
Urgency/impact phrases such as “critical” or “outage” map to High; issue phrases
such as “not working” or “cannot access” map to Medium; tickets without either
signal map to Low. Negated urgency cues are ignored. Review estimates against
your team's service-level policy before acting on them.

Run the priority-rule tests with:

```powershell
python -m unittest test_priority.py
```

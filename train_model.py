
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "all_tickets_processed_improved_v3.csv"
MODEL_PATH = BASE_DIR / "ticket_model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Document", "Topic_group"])
    df["Document"] = df["Document"].astype(str).str.strip()
    df = df[df["Document"].ne("")]

    if df["Topic_group"].nunique() < 2:
        raise ValueError("Training data must contain at least two ticket categories.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["Document"],
        df["Topic_group"],
        test_size=0.2,
        random_state=42,
        stratify=df["Topic_group"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=30000,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\nDataset size: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

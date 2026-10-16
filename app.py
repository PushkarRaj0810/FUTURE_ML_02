
import streamlit as st
import joblib
from pathlib import Path

from priority import classify_priority

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ticket_model.joblib"

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🎫 Support Ticket Classification")
st.write(
    "Classify customer support tickets and estimate their priority."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Trained model not found. Run `py train_model.py` first.")
    st.stop()

ticket = st.text_area(
    "Enter the support ticket:",
    placeholder="Example: I cannot access my account.",
    height=150,
)

if st.button("Classify Ticket", type="primary"):
    if not ticket.strip():
        st.warning("Please enter a ticket description.")
    else:
        category = model.predict([ticket])[0]
        priority, priority_reason = classify_priority(ticket)
        confidence = max(model.predict_proba([ticket])[0])

        st.subheader("Prediction Results")
        st.success(f"**Category:** {category}")
        st.caption(f"Category model confidence: {confidence:.0%}")

        if priority == "High":
            st.error(f"**Estimated Priority:** {priority}")
        elif priority == "Medium":
            st.warning(f"**Estimated Priority:** {priority}")
        else:
            st.info(f"**Estimated Priority:** {priority}")

        st.write(priority_reason)
        st.caption(
            "Category is predicted by the trained ML model. Priority is "
            "estimated from urgency and impact phrases; it is not a trained "
            "priority model. Confirm the result against your support team's "
            "service-level policy."
        )
